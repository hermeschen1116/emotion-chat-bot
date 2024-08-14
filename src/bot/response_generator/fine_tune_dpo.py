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
from unsloth import 
from trl import DPOConfig, DPOTrainer
from trl.core import LengthSampler
from unsloth import FastLanguageModel, PatchDPOTrainer

from libs import CommonScriptArguments, CommonWanDBArguments

PatchDPOTrainer() # not very sure what it does

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
	"Shotaro30678/rlhf-RG-trl-style-refined",
	split="train",
	keep_in_memory=True,
	num_proc=16,
	trust_remote_code=True
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
	low_cpu_mem_usage=True,
)
tokenizer.padding_side = "left"
tokenizer.clean_up_tokenization_spaces = True
tokenizer.chat_template = wandb.config["chat_template"]
tokenizer.add_special_tokens(wandb.config["special_tokens"])
base_model.resize_token_embeddings(len(tokenizer))

model = PeftModel.from_pretrained(
    base_model,
    wandb.config["adapter"],
    is_trainable=True,
    adapter_name="traingg",
    )

model.load_adapter(
    wandb.config["adapter"],
    adapter_name="reference"
    )

model.print_trainable_parameters()

training_args = DPOConfig(
    output_dir="./output",
    beta=0.1,
    model_adapter_name="traingg",
    ref_adapter_name="reference",
    gradient_checkpointing=True
)

dpo_trainer = DPOTrainer(
    model,
    # ref_model=None,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
)
dpo_trainer.train()

dpo_trainer.save_model("./dpo_v1")
