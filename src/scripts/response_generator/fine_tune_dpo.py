from argparse import ArgumentParser
from dataclasses import Field, dataclass

import wandb
from datasets import load_dataset
from peft.peft_model import PeftModel
from transformers import HfArgumentParser
from transformers.hf_argparser import HfArg
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer

from emotion_chat_bot.utils.CommonConfig import CommonScriptArguments, CommonWanDBArguments

PatchDPOTrainer()  # not very sure what it does


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
	resume=wandb_args.resume,
)
run.config["chat_template"] = chat_template["template"]
run.config["instruction_template"] = chat_template["instruction"]
run.config["response_template"] = chat_template["response"]
run.config["special_tokens"] = chat_template["special_tokens"]

# Load Dataset
dataset = load_dataset(
	run.config["dataset"],
	split="train",
	keep_in_memory=True,
	num_proc=16,
	trust_remote_code=True,
)

# Load Tokenizer
base_model, tokenizer = FastLanguageModel.from_pretrained(
	run.config["base_model"],
	attn_implementation="flash_attention_2",
	pretraining_tp=1,
	load_in_4bit=True,
	use_cache=False,
	device_map="auto",
	use_gradient_checkpointing="unsloth",
	low_cpu_mem_usage=True,
)
tokenizer.padding_side = "left"
tokenizer.clean_up_tokenization_spaces = True
tokenizer.chat_template = run.config["chat_template"]
tokenizer.add_special_tokens(run.config["special_tokens"])
base_model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(
	base_model,
	run.config["adapter"],
	is_trainable=True,
	adapter_name="training",
)

model.load_adapter(run.config["adapter"], adapter_name="reference")

model.print_trainable_parameters()

training_args = DPOConfig(
	output_dir="./checkpoints",
	beta=0.1,
	model_adapter_name="training",
	ref_adapter_name="reference",
	remove_unused_columns=False,
	num_train_epochs=3,
	gradient_checkpointing=True,
)

dpo_trainer = DPOTrainer(
	model,
	# ref_model=None,
	args=training_args,
	train_dataset=dataset,
	tokenizer=tokenizer,
)
dpo_trainer.train()

# 16-bit working, but 4-bit somehow not working
model.save_pretrained_merged(
	"16bit_model_3epo-v3",
	tokenizer,
	save_method="merged_16bit",
)
