from argparse import ArgumentParser
from dataclasses import dataclass, Field

import torch
import wandb
from bitsandbytes.optim import PagedLion32bit
from datasets import load_dataset
from peft.peft_model import PeftModel
from torch import tensor
from tqdm.auto import tqdm
from transformers import (GenerationConfig, HfArgumentParser, pipeline, TextStreamer)
from transformers.hf_argparser import HfArg
from trl import AutoModelForCausalLMWithValueHead
import pandas as pd
from trl.core import LengthSampler
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
	"hermeschen1116/daily_dialog_for_RG",
	split="train+validation",
	keep_in_memory=True,
	num_proc=16,
	trust_remote_code=True
)

history_length: int = 2 * wandb.config["num_turns_history"]
dataset = dataset.filter(lambda sample: len(sample) >= (2 + history_length), input_columns="prompt", num_proc=16)
print(f"Dataset size after filter: {len(dataset)}")
dataset = dataset.take(1024)   # use very small dataset to debug

dataset = dataset.map(lambda sample: {
	"prompt": sample[i: i + 2 + history_length] for i in range(0, len(sample) - 2, 2)
			  if (i + 2 + history_length) <= len(sample)
}, input_columns="prompt", num_proc=16)

system_prompt: list = [{"role": "system", "content": {"emotion": "", "dialog": wandb.config["system_prompt"]}}]

dataset = dataset.map(lambda samples: {
	"prompt": [system_prompt + sample for sample in samples]
}, input_columns="prompt", batched=True, num_proc=16)

emotion_labels: list = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]

dataset = dataset.map(lambda samples: {
	"query": [sample[:-1] +
			  [{"role": "assistant", "content": {"emotion": sample[-1]["content"]["emotion"], "dialog": ""}}]
	          for sample in samples],
	"label": [sample[-1]["content"]["emotion"] for sample in samples]
}, input_columns="prompt", remove_columns="prompt", batched=True, num_proc=16)

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
    num_workers=16,
    torch_dtype="auto",
    add_special_tokens=True,
    truncation=False,
    padding=True
)

dataset = dataset.with_format("torch")
dataset = dataset.map(lambda sample: {
	"input_ids": tokenizer.apply_chat_template(sample,
	                                           tokenize=True,
	                                           padding="max_length",
	                                           max_length=wandb.config["max_input_tokens"],
	                                           add_generation_prompt=True,
	                                           return_tensors="pt")
}, input_columns="query", num_proc=16)

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
 
def best_of_reward(batch: dict) -> list:
	best_of_emotion_scores = []
	best_of_length_scores = []
	best_of_gibberish_scores = []
	results = []
	
	# result of best_of emotion reward
	for responses, emotion in zip(batch["response_best_of"], batch["label"]):
		emotion_scores = [emotion_reward(response, emotion) for response in responses]
		best_of_emotion_scores.append(emotion_scores)
	# print(best_of_emotion_scores)
 
	# result of best_of length reward
	for response_lengths in batch["response_best_of_len"]:
		# print(response_lengths)
		length_scores = [length_reward(response_length) for response_length in response_lengths]
		best_of_length_scores.append(length_scores)
	# print(best_of_length_scores)
 
	# result of best_of non_gibberish reward
	for responses in batch["response_best_of"]:
		gibberish_scores = [non_gibberish_reward(response) for response in responses]
		best_of_gibberish_scores.append(gibberish_scores)
	# print(best_of_gibberish_scores)

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
device = 0 if torch.cuda.is_available() else "cpu"

# :: [Resp]
response_tensors_ref, response_tensors = [], []
response_tensors_ref_len, response_tensors_len = [], []
# :: [[Resp]]
response_tensors_best_of = []
response_tensors_best_of_len = []
input_ref = []

query_tensors = [input_ids.squeeze(0) for input_ids in dataset["input_ids"]]

for i in range(len(dataset)):
    query = query_tensors[i]
    input = tokenizer.decode(query, skip_special_tokens=True)
    input_len = len(input)
    input_ref.append(input)
    
    output = bot(
        input,
        **gen_kwargs,
        streamer=streamer
    )
    response = output[0]['generated_text'][input_len:]
    response_tensors_ref.append(response)
    response_tensors_ref_len.append(len(response))

    # generating copies of the same query for the Best-of-n sampling
    inputs = [input for _ in range(N_BEST_OF)]
    output = bot(
        inputs,
        **gen_kwargs,
        streamer=streamer
    )
    responses = [text[0]['generated_text'][input_len:] for text in output]
    response_tensors_best_of.append(responses)
    response_tensors_best_of_len.append([len(response) for response in responses])

test_data = (dataset.map(lambda sample: {
    "query": sample["query"],
    "label": sample["label"],
    "input_ids": sample["input_ids"],
})           
.add_column("input", input_ref)
.add_column("response_ref", response_tensors_ref)
.add_column("response_ref_len", response_tensors_ref_len)
.add_column("response_best_of", response_tensors_best_of)
.add_column("response_best_of_len", response_tensors_best_of_len))

scores_best_of = [torch.tensor(reward) for reward in best_of_reward(test_data)]

response_best_of = [response_tensors_best_of[i][a.argmax().item()] for i, a in enumerate(scores_best_of)]
response_best_of_reject = [response_tensors_best_of[i][a.argmin().item()] for i, a in enumerate(scores_best_of)]

output_dataset = dataset.remove_columns(["input_ids"])
output_dataset = (
    output_dataset
    .add_column("prompt", input_ref)
    .add_column("chosen", response_best_of)
    .add_column("chosen_score", [scores.max().item() for scores in scores_best_of])
    .add_column("rejected", response_best_of_reject)
    .add_column("rejected_score", [scores.min().item() for scores in scores_best_of])
)

output_dataset.push_to_hub("Shotaro30678/rlhf-RG-trl-style-raw-1024-query")