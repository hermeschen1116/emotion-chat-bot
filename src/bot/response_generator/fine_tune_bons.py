from argparse import ArgumentParser
from dataclasses import dataclass, Field

import torch
import wandb
from bitsandbytes.optim import PagedLion32bit
from datasets import load_dataset
from peft.peft_model import PeftModel
from torch import tensor
from tqdm.auto import tqdm
from transformers import (BitsAndBytesConfig, GenerationConfig, HfArgumentParser, pipeline, TextStreamer,AutoModelForSequenceClassification)
from transformers.hf_argparser import HfArg
from trl import AutoModelForCausalLMWithValueHead
from trl.trainer.rloo_trainer import  RLOOConfig, RLOOTrainer
from trl.core import LengthSampler
from trl.extras import BestOfNSampler
from unsloth import FastLanguageModel

from libs import CommonScriptArguments, CommonWanDBArguments

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
dataset = dataset.take(5)   # use very small dataset to debug

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

ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model_with_adapter, device_map="auto")

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
		# "quantization_config": BitsAndBytesConfig(
		# 	load_in_4bit=True,
		# 	bnb_4bit_compute_dtype=torch.float16
		# ),
		"id2label": {k: v for k, v in enumerate(emotion_labels)},
		"label2id": {v: k for k, v in enumerate(emotion_labels)},
		"low_cpu_mem_usage": True
	},
	trust_remote_code=True
)

# # Detect gibberish
# gibberish_analyser = pipeline(
# 	model=wandb.config["gibberish_detector_model"],
# 	tokenizer=wandb.config["gibberish_detector_model"],
# 	max_length=512,
# 	framework="pt",
# 	task="text-classification",
# 	num_workers=16,
# 	device_map="cpu",
# 	torch_dtype="auto",
# 	model_kwargs={
# 		# "quantization_config": BitsAndBytesConfig(
# 		# 	load_in_4bit=True,
# 		# 	bnb_4bit_compute_dtype=torch.float16
# 		# ),
# 		"low_cpu_mem_usage": True
# 	},
# 	trust_remote_code=True
# )


# def emotion_reward(response: str, emotion: str) -> float:
# 	score = sentiment_analyser(response)[0]

# 	if score["label"] == emotion:
# 		return score["score"] * 10
# 	else:
# 		return score["score"] - 1


# def non_gibberish_reward(response: str) -> float:
# 	score = gibberish_analyser(response)[0]

# 	match score["label"]:
# 		case "clean":
# 			return score["score"] * 10
# 		case "mild gibberish":
# 			return score["score"] * 0.9
# 		case _:
# 			return score["score"] - 1


# # [TODO] 用級距的方式來給予分數
# def length_reward(response_length: int) -> float:
# 	difference_ratio_min = (response_length - wandb.config["min_new_tokens"]) / wandb.config["min_new_tokens"]
# 	difference_ratio_max = (response_length - wandb.config["max_new_tokens"]) / wandb.config["max_new_tokens"]

# 	if abs(difference_ratio_min) < 1:
# 		return difference_ratio_min * 0.0001
# 	elif abs(difference_ratio_min) > 1 > abs(difference_ratio_max):
# 		return abs(difference_ratio_min + difference_ratio_max) * 10
# 	else:
# 		return difference_ratio_max * 0.9


# def reward(batch: dict) -> list:
# 	emotion_scores = [emotion_reward(response, emotion)
# 	                  for response, emotion in zip(batch["response"], batch["label"])]
# 	length_scores = [length_reward(response_length) for response_length in batch["response_length"]]
# 	gibberish_scores = [non_gibberish_reward(response) for response in batch["response"]]

# 	reward_weight = tensor(wandb.config["reward_weights"], dtype=torch.float)
# 	reward_bias = tensor(wandb.config["reward_bias"], dtype=torch.float)
# 	return [reward_weight.dot(tensor(reward_score, dtype=torch.float)) + reward_bias
# 	        for reward_score in zip(emotion_scores, length_scores, gibberish_scores)]

# [TODO] somehow if I comment this it stops working
optimizer = PagedLion32bit(filter(lambda p: p.requires_grad, ppo_model.parameters()), lr=wandb.config["max_new_tokens"])
# lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=wandb.config["lr_gamma"])
length_sampler = LengthSampler(wandb.config["min_new_tokens"], wandb.config["max_new_tokens"])

streamer = TextStreamer(
	tokenizer,
	skip_special_tokens=True,  # show <pad> or not
	clean_up_tokenization_spaces=True
)

generation_config = GenerationConfig(
	max_length=None,
	min_length=-1,
	top_k=wandb.config["top_k"],
	top_p=wandb.config["top_p"],
	do_sample=True,
	use_cache=True,
	repetition_penalty=wandb.config["repetition_penalty"],
	pad_token_id=tokenizer.pad_token_id,
	bos_token_id=tokenizer.bos_token_id,
	eos_token_id=tokenizer.eos_token_id,
	low_memory=True
)

N_BEST_OF = 4
device = 0 if torch.cuda.is_available() else "cpu"

# :: [Resp]
response_tensors_ref, response_tensors = [], []
response_tensors_ans = []
# :: [[Resp]]
response_tensors_best_of = []

gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id}

query_tensors = [input_ids.squeeze(0) for input_ids in dataset["input_ids"]]

for i in range(len(dataset)):
    gen_len = length_sampler()
    query = query_tensors[i]
    input = tokenizer.decode(query)
    
    output = ppo_model.generate(
        query.unsqueeze(dim=0).to(device),
        max_new_tokens=gen_len,
        generation_config=generation_config,
		streamer=streamer,  # use streamer to show the generation process
        ).squeeze()
    response_tensors_ref.append(tokenizer.decode(output)[len(input):])
    

    # output = model.generate(query.unsqueeze(dim=0).to(device), max_new_tokens=gen_len, **gen_kwargs).squeeze()
    # response_tensors.append(tokenizer.decode(output))

    # generating copies of the same query for the Best-of-n sampling
    queries = query.repeat((N_BEST_OF, 1))
    output = ppo_model.generate(
        queries.to(device),
        max_new_tokens=gen_len,
        generation_config=generation_config,
        # batch_size=1,
		# streamer=streamer,  # use streamer to show the generation process
        ).squeeze()
    response_tensors_best_of.append([text[len(input):] for text in tokenizer.batch_decode(output, skip_special_tokens=False)]
)
    
scores_best_of = []
label_best_of = []
for i, response in enumerate(response_tensors_best_of):
    # print(response)
    # print(sentiment_analyser(response))
    for output in sentiment_analyser(response):
        print(output)
    print()
    # base_score = scores_ref[i]
    scores_best_of.append(torch.tensor([output["score"] for output in sentiment_analyser(response)]))
    # label_best_of.append(torch.tensor([output["label"] for output in sentiment_analyser(response)]))
  
scores_ref = [output["score"] for output in sentiment_analyser(response_tensors_ref)]
emotion_ref = [output["label"] for output in sentiment_analyser(response_tensors_ref)]  
# print(dataset['label'])
# label = [emotion_labels[emo] for emo in dataset['label']]

output_data = dict()
output_data["query"] = dataset["query"]
import pandas as pd
output_data["label"] = dataset['label']
output_data["response (ref)"] = response_tensors_ref
output_data["emotion (ref)"] = emotion_ref
output_data["scores (ref)"] = scores_ref
output_data["response (best_of)"] = [
    response_tensors_best_of[i][a.argmax().item()] for i, a in enumerate(scores_best_of)
]
output_data["scores (best_of)"] = [a.max().item() for a in scores_best_of]
df_results = pd.DataFrame(output_data)
print(df_results)