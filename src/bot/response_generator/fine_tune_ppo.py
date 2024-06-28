import tempfile
from argparse import ArgumentParser
from dataclasses import dataclass

from pyarrow import Field
import torch
import wandb
from datasets import concatenate_datasets, load_dataset
from lion_pytorch import Lion
from peft.peft_model import PeftModel
from tqdm.auto import tqdm
from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
	BitsAndBytesConfig,
	GenerationConfig,
	HfArgumentParser,
	TextClassificationPipeline
)
from transformers.hf_argparser import HfArg
from trl import AutoModelForCausalLMWithValueHead, DataCollatorForCompletionOnlyLM, PPOConfig, PPOTrainer
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
	name=wandb_args.name,
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
dataset = load_dataset("hermeschen1116/daily_dialog_for_RG", num_proc=16, trust_remote_code=True)
dataset = concatenate_datasets([dataset["train"], dataset["validation"]])
# dataset = dataset.train_test_split(train_size=0.001)["train"]

system_prompt: list = [{"role": "system", "content": {"emotion": "", "dialog": wandb.config["system_prompt"]}}]

dataset = dataset.map(lambda samples: {
	"prompt": [system_prompt + sample for sample in samples]
}, input_columns="prompt", batched=True, num_proc=16)

dataset = dataset.map(lambda samples: {})

# Load Tokenizer
base_model, tokenizer = FastLanguageModel.from_pretrained(
	wandb.config["tokenizer"],
	attn_implementation="flash_attention_2",
	pretraining_tp=1,
	load_in_4bit=(wandb.config["init_lora_weights"] != "loftq"),
	quant_method="quantized",
	use_cache=False,
	device_map="auto",
	use_gradient_checkpointing=True,
	low_cpu_mem_usage=True,
	trust_remote_code=True,
)
tokenizer.padding_side = "left"
tokenizer.clean_up_tokenization_spaces = True
tokenizer.chat_template = wandb.config["chat_template"]
tokenizer.add_special_tokens(wandb.config["special_tokens"])
base_model.resize_token_embeddings(len(tokenizer))

base_model = PeftModel.from_pretrained(base_model, run.use_model(wandb.config["base_model"]))
base_model.print_trainable_parameters()
FastLanguageModel.for_training(base_model)

base_model = AutoModelForCausalLMWithValueHead.from_pretrained(
	base_model
)

sentiment_analysis_model = AutoModelForSequenceClassification.from_pretrained(
	wandb.config["sentiment_analysis_model"],
	quantization_config=BitsAndBytesConfig(
		load_in_4bit=True,
		bnb_4bit_compute_dtype=torch.float16
	),
	device_map="auto",
	low_cpu_mem_usage=True
)

sentiment_analysis_tokenizer = AutoTokenizer.from_pretrained(
	wandb.config["sentiment_analysis_tokenizer"],
	trust_remote_code=True
)

analyser = TextClassificationPipeline(
	model=sentiment_analysis_model,
	tokenizer=sentiment_analysis_tokenizer,
	framework="pt",
	task="sentiment-analysis",
	num_workers=16,
	torch_dtype="auto"
)

sentiment_analysis_model = torch.compile(sentiment_analysis_model)

# [TODO] a reward function contain length and emotion
def reward():
	return 1


ppo_config = PPOConfig(
	gradient_accumulation_steps=1,
	learning_rate=wandb.config["learning_rate"],
	max_grad_norm=wandb.config["max_grad_norm"],
	log_with="wandb",
	optimize_cuda_cache=True,
	optimize_device_cache=True,
	early_stopping=True,
	is_peft_model=True,
	use_score_scaling=True,
	use_score_norm=True,
	score_clip=wandb.config["score_clip"],
)

special_tokens_map: dict = dict(zip(tokenizer.all_special_tokens, [[ids] for ids in tokenizer.all_special_ids]))
data_collator = DataCollatorForCompletionOnlyLM(
	special_tokens_map[wandb.config["response_template"]],
	instruction_template=special_tokens_map[wandb.config["instruction_template"]],
	tokenizer=tokenizer
)
optimizer = Lion(filter(lambda p: p.requires_grad, base_model.parameters()), lr=ppo_config.learning_rate)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

generation_config = GenerationConfig(
	min_length=wandb.config["min_length"],
	top_k=wandb.config["top_k"],
	top_p=wandb.config["top_p"],
	do_sample=True,
	max_new_tokens=wandb.config["max_new_tokens"],
	repetition_penalty=wandb.config["repetition_penalty"],
	pad_token_id=tokenizer.pad_token_id,
	eos_token_id=tokenizer.eos_token_id
)

# Setup Tuner
tuner = PPOTrainer(
	config=ppo_config,
	model=base_model,
	data_collator=data_collator,
	tokenizer=tokenizer,
	dataset=dataset["train"],
	optimizer=optimizer,
	lr_scheduler=lr_scheduler
)

for epoch in tqdm(range(wandb.config["num_epochs"]), "epoch: "):
	for batch in tqdm(tuner.dataloader):
		query_tensors = batch["input_ids"]

		# Get response from SFTModel
		response_tensors = tuner.generate(
			query_tensors,
			return_prompt=False,
			generation_config=generation_config
		)
		batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

		# Compute reward score
		pipe_outputs = reward(batch)
		rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

		# Run PPO step
		stats = tuner.step(query_tensors, response_tensors, rewards)
		tuner.log_stats(stats, batch, rewards)

# model_artifact = wandb.Artifact(
# 	wandb.config["fine_tuned_model"],
# 	type="model"
# )

tuner.model = torch.compile(tuner.model)
tuner.model.push_to_hub(repo_id="response_generator_for_emotion_chat_bot", commit="", create_pr=True)
# with tempfile.TemporaryDirectory() as temp_dir:
# 	tuner.model.save_pretrained(temp_dir, save_embedding_layers=True)
# 	model_artifact.add_dir(temp_dir)
# 	run.log_artifact(model_artifact)

wandb.finish()
