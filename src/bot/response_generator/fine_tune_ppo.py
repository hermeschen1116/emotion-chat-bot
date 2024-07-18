from argparse import ArgumentParser
from dataclasses import dataclass, Field

import torch
from torch import tensor
from trl.core import LengthSampler
import wandb
from bitsandbytes.optim import PagedLion32bit
from datasets import load_dataset
from peft.peft_model import PeftModel
from tqdm.auto import tqdm
from transformers import (
	BitsAndBytesConfig,
	GenerationConfig,
	HfArgumentParser,
	pipeline, TextStreamer
)
from transformers.hf_argparser import HfArg
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
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
# dataset = dataset.take(2048)   # use very small dataset to debug

history_length: int = 2 * wandb.config["num_turns_history"]
dataset = dataset.filter(lambda sample: len(sample) >= (2 + history_length), input_columns="prompt", num_proc=16)
print(f"Dataset size after filter: {len(dataset)}")

dataset = dataset.map(lambda sample: {
	"prompt": sample[i: i + 2 + history_length]
	for i in range(0, len(sample) - 2, 2) if (i + 2 + history_length) <= len(sample)
}, input_columns="prompt", num_proc=16)

system_prompt: list = [{"role": "system", "content": {"emotion": "", "dialog": wandb.config["system_prompt"]}}]

dataset = dataset.map(lambda samples: {
	"prompt": [system_prompt + sample for sample in samples]
}, input_columns="prompt", batched=True, num_proc=16)

emotion_labels: list = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]

dataset = dataset.map(lambda samples: {
	"query": [
		sample[:-1] + [{"role": "assistant", "content": {"emotion": sample[-1]["content"]["emotion"], "dialog": ""}}]
		for sample in samples
	],
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
	low_cpu_mem_usage=True,
)
tokenizer.padding_side = "left"
tokenizer.clean_up_tokenization_spaces = True
tokenizer.chat_template = wandb.config["chat_template"]
tokenizer.add_special_tokens(wandb.config["special_tokens"])
base_model.resize_token_embeddings(len(tokenizer))

base_model_with_adapter = PeftModel.from_pretrained(base_model, wandb.config["adapter"])
base_model_with_adapter.print_trainable_parameters()
FastLanguageModel.for_inference(base_model_with_adapter)

ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
	base_model_with_adapter,
	device_map="auto"
)

dataset = dataset.with_format("torch")
dataset = dataset.map(lambda sample: {
	"input_ids": tokenizer.apply_chat_template(
		sample,
		tokenize=True,
		padding="max_length",
		max_length=wandb.config["max_input_tokens"],
		add_generation_prompt=True,
		return_tensors="pt"
	)
}, input_columns="query", num_proc=16)

# Sentiment Analysis
analyser = pipeline(
	model=wandb.config["sentiment_analysis_model"],
 	tokenizer=wandb.config["sentiment_analysis_model"],
	max_length=512,
	truncation=True,
	framework="pt",
	task="sentiment-analysis",
	num_workers=16,
	device_map="auto",
	torch_dtype="auto",
	model_kwargs={
		"quantization_config": BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_compute_dtype=torch.float16
		),
		"id2label": {k: v for k, v in enumerate(emotion_labels)},
		"label2id": {v: k for k, v in enumerate(emotion_labels)},
		"low_cpu_mem_usage": True
	},
	trust_remote_code=True
)

sentiment_analysis_model = torch.compile(analyser.model)


def emotion_reward(response: str, emotion: str) -> float:
    emotion_score = analyser(response)

    if emotion_score["label"] == emotion:
        return emotion_score["score"] * 10
    else:
        return emotion_score["score"] - 1


# [TODO] 用級距的方式來給予分數
def length_reward(response_length: int) -> float:
	difference_ratio_min = (response_length - wandb.config["min_new_tokens"]) / wandb.config["min_new_tokens"]
	difference_ratio_max = (response_length - wandb.config["max_new_tokens"]) / wandb.config["max_new_tokens"]

	if difference_ratio_min < 1:
		return difference_ratio_min * 0.0001
	elif difference_ratio_min > 1 and difference_ratio_max < 1:
		return (difference_ratio_min + difference_ratio_max) * 10
	elif difference_ratio_max > 1:
		return difference_ratio_max * 0.9


def reward(batch: dict) -> list:
    emotion_reward_scores = [emotion_reward(response, emotion) for response, emotion in zip(batch["response"], batch["label"])]
    length_reward_scores = [length_reward(response_length) for response_length in batch["response_length"]]

    reward_weights = tensor(wandb.config["reward_weights"])
    reward_bias = tensor(wandb.config["reward_bias"])
    return [tensor(reward) * reward_weights + reward_bias for reward in zip(emotion_reward_scores, length_reward_scores)]

ppo_config = PPOConfig(
	gradient_accumulation_steps=1,
	learning_rate=wandb.config["learning_rate"],
	max_grad_norm=wandb.config["max_grad_norm"],
	log_with="wandb",
	optimize_device_cache=True,
	early_stopping=True,
	is_peft_model=True,
	use_score_scaling=True,
	use_score_norm=True,
	score_clip=wandb.config["score_clip"],
	# gradient_checkpointing=True
)

optimizer = PagedLion32bit(filter(lambda p: p.requires_grad, base_model.parameters()), lr=ppo_config.learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=wandb.config["lr_gamma"])
length_sampler = LengthSampler(wandb.config["min_new_tokens"], wandb.config["max_new_tokens"])

streamer = TextStreamer(
	tokenizer,
	skip_special_tokens=True, # show <pad> or not
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

# Setup Tuner
tuner = PPOTrainer(
	config=ppo_config,
	model=ppo_model,
	tokenizer=tokenizer,
	dataset=dataset,
	optimizer=optimizer,
	lr_scheduler=lr_scheduler
)

for epoch in range(wandb.config["num_epoches"]):
	for batch in tqdm(tuner.dataloader, desc=f"epoch{epoch}", colour="yellow"):
		query_tensors = [input_ids.squeeze(0) for input_ids in batch["input_ids"]]

		response_tensors = tuner.generate(
			query_tensors,
			return_prompt=False,
			batch_size=1,   # must set to 1 if using streamer
			streamer=streamer,  # use streamer to show the generation process
			length_sampler=length_sampler,
			**generation_config.to_dict()
		)
		batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
		batch["response_length"] = [r.size()[0] for r in response_tensors]
		response_tensors = [torch.LongTensor(t.to("cpu")) for t in response_tensors]

		reward_scores = reward(batch)
		rewards = [torch.FloatTensor(scores.to("cpu")) for scores in reward_scores]

		stats = tuner.step(query_tensors, response_tensors, rewards)
		tuner.log_stats(stats, batch, rewards)

tuner.model = torch.compile(tuner.model)
tuner.model.push_to_hub(repo_id="response_generator_for_emotion_chat_bot", commit="", create_pr=True)

wandb.finish()
