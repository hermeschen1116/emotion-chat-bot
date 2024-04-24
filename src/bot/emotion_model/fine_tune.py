import os
from dataclasses import dataclass
from typing import Any

import huggingface_hub
import torch
import wandb
from datasets import load_from_disk
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, HfArgumentParser
from trl import SFTTrainer

from libs.CommonConfig import CommonWanDBArguments, CommonScriptArguments


@dataclass
class ScriptArguments(CommonScriptArguments):
    dtype: Any
    device: str


parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments, BitsAndBytesConfig))
args, wandb_args, quantization_config, remain_args = parser.parse_args()

load_dotenv(encoding="utf-8")
huggingface_hub.login(token=os.environ.get("HF_TOKEN", args.huggingface_api_token), add_to_git_credential=True)
wandb.login(key=os.environ.get("WANDB_API_KEY", args.wandb_api_token), relogin=True)

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

# Load Dataset
dataset_path = run.use_artifact("daily_dialog_for_RG_train:latest").download()
dataset = load_from_disk(dataset_path)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model_name, truncation=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix for fp16
tokenizer.truncation_side = "right"

# Configurations
quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                         bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=False)
quantization_config = quantization_config if torch.cuda.is_available() else None

peft_parameters = LoraConfig(lora_alpha=16, lora_dropout=0.1, r=8, bias="none", task_type="CAUSAL_LM")

num_train_epochs: int = 1

train_params = TrainingArguments(output_dir=f"./checkpoints_{fine_tuned_model_name}", num_train_epochs=num_train_epochs,
                                 per_device_train_batch_size=4, gradient_accumulation_steps=1,
                                 optim="paged_adamw_32bit", save_steps=25, logging_steps=25, learning_rate=2e-4,
                                 weight_decay=0.001, fp16=False, bf16=False, max_grad_norm=0.3, max_steps=-1,
                                 warmup_ratio=0.03, group_by_length=True, lr_scheduler_type="constant",
                                 report_to=["tensorboard"], gradient_checkpointing=True,
                                 gradient_checkpointing_kwargs={"use_reentrant": True})

# Load Model
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, quantization_config=quantization_config,
                                                  device_map="auto" if torch.cuda.is_available() else "cpu",
                                                  low_cpu_mem_usage=True)

base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# Setup Tuner
tuner = SFTTrainer(model=base_model, train_dataset=train_data, dataset_text_field="prompt", tokenizer=tokenizer,
                   peft_config=peft_parameters, args=train_params, max_seq_length=1024)

tuner.train()

tuner.model.save_pretrained(fine_tuned_model_name)
