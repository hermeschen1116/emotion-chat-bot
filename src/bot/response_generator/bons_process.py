from argparse import ArgumentParser
from dataclasses import dataclass, Field
import os

import pickle
import numpy as np

import torch
import wandb
from datasets import load_dataset, Dataset
from peft.peft_model import PeftModel
from torch import tensor
from tqdm.auto import tqdm
from transformers import (HfArgumentParser, pipeline, TextStreamer)
from transformers.hf_argparser import HfArg
import pandas as pd
from unsloth import FastLanguageModel

from libs import CommonScriptArguments, CommonWanDBArguments, ResponseGeneratorPipeline

@dataclass
class ScriptArguments(CommonScriptArguments):
	chat_template_file: Field[str] = HfArg(aliases="--chat-template-file", default="")


config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)

chat_template: dict = eval(open(args.chat_template_file, "r", encoding="utf-8", closefd=True).read())

# Initialize Wandb
run = wandb.init(
	job_type=wandb_args.job_type,
	config=wandb_args.config,
	project=wandb_args.project,
	group=wandb_args.group,
	notes=wandb_args.notes,
	mode=wandb_args.mode,
	resume=wandb_args.resume
)
wandb.config["chat_template"] = chat_template["template"]
wandb.config["instruction_template"] = chat_template["instruction"]
wandb.config["response_template"] = chat_template["response"]
wandb.config["special_tokens"] = chat_template["special_tokens"]

# Load Dataset
dataset = load_dataset(
	wandb.config["dataset"],
	split="train",
	keep_in_memory=True,
	num_proc=16,
	trust_remote_code=True
)
dataset = dataset.take(1024)  

target_score_range = wandb.config["target_score_range"]
# If score diff < 5 than flag 
def mark_difference(example):
    return 1 if (example["chosen_score"] - example["rejected_score"]) < target_score_range else 0

dataset = (
    dataset
    .add_column("flag", [mark_difference(example) for example in dataset])
    # flag=1 means score too close
)

# Load Tokenizer
base_model, tokenizer = FastLanguageModel.from_pretrained(
	wandb.config["base_model"],
	attn_implementation="flash_attention_2",
	pretraining_tp=1,
	load_in_4bit=True,
	use_cache=False,
	device_map="auto",
	use_gradient_checkpointing=True,
	low_cpu_mem_usage=True
)
tokenizer.padding_side = "left"
tokenizer.clean_up_tokenization_spaces = True
tokenizer.chat_template = wandb.config["chat_template"]
tokenizer.add_special_tokens(wandb.config["special_tokens"])
base_model.resize_token_embeddings(len(tokenizer))

base_model_with_adapter = PeftModel.from_pretrained(base_model, wandb.config["adapter"])
base_model_with_adapter.print_trainable_parameters()
FastLanguageModel.for_inference(base_model_with_adapter)

bot = ResponseGeneratorPipeline(
    base_model_with_adapter,
    tokenizer,
    framework="pt",
    task="conversation-generation",
    num_workers=1, # cause some issue here
    torch_dtype="auto",
    add_special_tokens=True,
    truncation=False,
    padding=True
)

emotion_labels: list = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]

# Sentiment Analysis
sentiment_analyser = pipeline(
	model=wandb.config["sentiment_analysis_model"],
	tokenizer=wandb.config["sentiment_analysis_model"],
	max_length=512,
	framework="pt",
	task="sentiment-analysis",
	num_workers=16,
	device_map="cpu",
	torch_dtype="auto",
	model_kwargs={
		"id2label": {k: v for k, v in enumerate(emotion_labels)},
		"label2id": {v: k for k, v in enumerate(emotion_labels)},
		"low_cpu_mem_usage": True
	},
	trust_remote_code=True
)

# Detect gibberish
gibberish_analyser = pipeline(
	model=wandb.config["gibberish_detector_model"],
	tokenizer=wandb.config["gibberish_detector_model"],
	max_length=512,
	framework="pt",
	task="text-classification",
	num_workers=16,
	device_map="cpu",
	torch_dtype="auto",
	model_kwargs={
		"low_cpu_mem_usage": True
	},
	trust_remote_code=True
)

def emotion_reward(response: str, emotion: str) -> float:
	score = sentiment_analyser(response)[0]
 
	if score["label"] == emotion:
		return score["score"] * 10
	else:
		return score["score"] - 1


def non_gibberish_reward(response: str) -> float:
	score = gibberish_analyser(response)[0]

	match score["label"]:
		case "clean":
			return score["score"] * 10
		case "mild gibberish":
			return score["score"] * 0.9
		case _:
			return score["score"] - 1


# [TODO] 用級距的方式來給予分數
def length_reward(response_length: int) -> float:
	difference_ratio_min = (response_length - 5) / 5
	difference_ratio_max = (response_length - 20) / 20
 
	if abs(difference_ratio_min) < 1:
		return difference_ratio_min * 0.0001
	elif abs(difference_ratio_min) > 1 > abs(difference_ratio_max):
		return abs(difference_ratio_min + difference_ratio_max) * 10
	else:
		return difference_ratio_max * 0.9

def reward(batch: dict) -> list:
	emotion_scores = [emotion_reward(response, emotion)
	                  for response, emotion in zip(batch["response"], batch["label"])]
	length_scores = [length_reward(response_length) for response_length in batch["response_length"]]
	gibberish_scores = [non_gibberish_reward(response) for response in batch["response"]]

	reward_weight = tensor(wandb.config["reward_weights"], dtype=torch.float)
	reward_bias = tensor(wandb.config["reward_bias"], dtype=torch.float)
	return [reward_weight.dot(tensor(reward_score, dtype=torch.float)) + reward_bias
	        for reward_score in zip(emotion_scores, length_scores, gibberish_scores)]
 
def best_of_reward(batch: dict) -> list:
	best_of_emotion_scores = []
	best_of_length_scores = []
	best_of_gibberish_scores = []
	results = []
	
	# result of best_of emotion reward
	for responses, emotion in zip(batch["response_best_of"], batch["label"]):
		emotion_scores = [emotion_reward(response, emotion) for response in responses]
		best_of_emotion_scores.append(emotion_scores)
 
	# result of best_of length reward
	for response_lengths in batch["response_best_of_len"]:
		# print(response_lengths)
		length_scores = [length_reward(response_length) for response_length in response_lengths]
		best_of_length_scores.append(length_scores)
 
	# result of best_of non_gibberish reward
	for responses in batch["response_best_of"]:
		gibberish_scores = [non_gibberish_reward(response) for response in responses]
		best_of_gibberish_scores.append(gibberish_scores)

	reward_weight = tensor(wandb.config["reward_weights"], dtype=torch.float)
	reward_bias = tensor(wandb.config["reward_bias"], dtype=torch.float)
	for reward_scores in zip(best_of_emotion_scores, best_of_length_scores, best_of_gibberish_scores):
		reward_score = [list(tup) for tup in zip(*reward_scores)]
		result = [reward_weight.dot(tensor(reward_s, dtype=torch.float)) + reward_bias for reward_s in reward_score]
		results.append(result)
	return results

streamer = TextStreamer(
    tokenizer,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=True
)

gen_kwargs = {
    "max_new_tokens":wandb.config["max_new_tokens"],
    "min_new_tokens":wandb.config["min_new_tokens"],
    "repetition_penalty":wandb.config["repetition_penalty"],
    "top_k":2,
    "top_p":1.0, 
    "temperature":30.0,
    "pad_token_id":tokenizer.pad_token_id,
	"eos_token_id":tokenizer.eos_token_id
}

N_BEST_OF = 5

# :: [Resp]
response_tensors_ref, response_tensors = [], []
response_tensors_ref_len, response_tensors_len = [], []
# :: [[Resp]]
response_tensors_best_of = []
response_tensors_best_of_len = []
input_ref = []



def calculate_score_diff(chosen_score, rejected_score):
    return chosen_score - rejected_score

def should_stop_regeneration(scores):
    median_score = np.median(scores)
    mean_score = np.mean(scores)
    print(f"current Median: {median_score:.3f}, Current Mean: {mean_score:.3f}\nDiff: {median_score-mean_score:.3f}")
    return abs(median_score - mean_score) < 0.5

# Function to save and load checkpoints
def save_checkpoint(data, scores, index, filename="checkpoint.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump((data, scores, index), f)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename="checkpoint.pkl"):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            data, scores, index = pickle.load(f)
        print(f"Checkpoint loaded from {filename}")
        return data, scores, index
    else:
        print(f"No checkpoint found in {filename}")
        return None, None, None

checkpoint_filename = "checkpoint.pkl"

# Try to load from checkpoint
updated_data, updated_scores, start_index = load_checkpoint(checkpoint_filename)
if updated_data is None or updated_scores is None or start_index is None:
    print("No checkpoint found, starting from scratch")
    # If no checkpoint, initialize from scratch
    updated_data = {
        'prompt': [data['prompt'] for data in dataset],
        'chosen': [data['chosen'] for data in dataset],
        'rejected': [data['rejected'] for data in dataset],
        'chosen_score': [data['chosen_score'] for data in dataset],
        'rejected_score': [data['rejected_score'] for data in dataset]
    }
    updated_scores = [calculate_score_diff(data['chosen_score'], data['rejected_score']) for data in dataset]
    start_index = 0

# Get original median and mean
original_scores = [calculate_score_diff(data['chosen_score'], data['rejected_score']) for data in dataset]
original_median = np.median(original_scores)
original_mean = np.mean(original_scores)

print(f"Original Median: {original_median:.3f}, Original Mean: {original_mean:.3f}")

# Initialize updated scores (diff)
if start_index == 0:
    updated_scores = original_scores.copy()

for i in range(start_index, len(dataset)):
    data = dataset[i]
    median_score = np.median(updated_scores)
    mean_score = np.mean(updated_scores)
    save_checkpoint(updated_data, updated_scores, i, checkpoint_filename)
    target_score_range = wandb.config["target_score_range"]
    
    # If the difference between median and mean is less than 0.5, stop generating
    if abs(median_score - mean_score) < 0.5:
        break
    
    if data['flag'] == 1:
        print(f"\nredo: {i}/{len(dataset)}")
        input_text = data['prompt']
        input_len = len(input_text)
        input_ref.append(input_text)

        inputs = [input_text for _ in range(N_BEST_OF)]
        
        fail_counter = 0
        
        while True:
            
            
            output = bot(inputs, **gen_kwargs)
            responses = [text[0]['generated_text'][input_len:] for text in output]
            tmp = {
                'response': responses,
                'response_length': [len(response) for response in responses],
                'label': data['label'],
            }
            score_tmp = [reward.item() for reward in reward(tmp)]  # Use item() to get Python scalar
            tmp["score"] = score_tmp
            score_range = max(score_tmp) - min(score_tmp)
            
            # If the generated output score range is less than expected, regenerate
            if score_range < target_score_range:
                fail_counter += 1
                print(f"fail: {fail_counter}/{10}")
                if fail_counter <= 10:
                    continue
                elif 10 < fail_counter <= 20:
                    target_score_range = 4
                    continue
                elif 20 < fail_counter:
                    target_score_range = 3
                    continue
            # Update score(diff)
            updated_scores[i] = calculate_score_diff(max(score_tmp), min(score_tmp))
            
            # Print out some info for reference
            print(f"\nLabel: {tmp['label']}\n")
            print(f"assistant: {dataset[i]['query'][4]['content']['dialog']}")
            print(f"user: {dataset[i]['query'][5]['content']['dialog']}\n")
                
            # Print output
            for j in range(N_BEST_OF):
                print(f"Score {j}: {score_tmp[j]:.3f}, Response: {tmp['response'][j]}")
                               
            print(f"\nRange of scores: {score_range:.3f}")
            print(f"\nchosen : {tmp['response'][score_tmp.index(max(score_tmp))]}")
            print(f"rejected : {tmp['response'][score_tmp.index(min(score_tmp))]}\n")
            median_score = np.median(updated_scores)
            mean_score = np.mean(updated_scores)
            print(f"Current Median: {median_score:.3f}, Current Mean: {mean_score:.3f}\nDiff: {abs(median_score - mean_score):.3f}")

            user_input = input("[y] accept, [else] decline: ").strip().lower()
            fail_counter = 0
            # If accepted, update updated_data
            if user_input == "y":
                updated_data['prompt'][i] = data['prompt']
                updated_data['chosen'][i] = tmp['response'][score_tmp.index(max(score_tmp))]
                updated_data['rejected'][i] = tmp['response'][score_tmp.index(min(score_tmp))]
                updated_data['chosen_score'][i] = max(score_tmp)
                updated_data['rejected_score'][i] = min(score_tmp)
                break
            else:
                continue
    else:
        print(f"\nskip: {i}/{len(dataset)}")
updated_dataset = Dataset.from_dict(updated_data)

# Show median and mean
final_scores = [calculate_score_diff(chosen, rejected) for chosen, rejected in zip(updated_data['chosen_score'], updated_data['rejected_score'])]
final_median = np.median(final_scores)
final_mean = np.mean(final_scores)
print(f"\nOriginal Median: {original_median:.3f}, Original Mean: {original_mean:.3f}")
print(f"Final Median: {final_median:.3f}, Final Mean: {final_mean:.3f}")

# Convert updated_data back to dataset format
updated_dataset = Dataset.from_dict(updated_data)
updated_dataset.push_to_hub("Shotaro30678/rlhf-RG-trl-style-picked")