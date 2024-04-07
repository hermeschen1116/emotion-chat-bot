import argparse
import os

import huggingface_hub
import torch
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

load_dotenv(encoding="utf-8")
huggingface_hub.login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=True)
wandb.login(key=os.environ.get("WANDB_API_KEY", ""), relogin=True)

# commandline inputs
parser = argparse.ArgumentParser(prog="Evaluation", description="Evaluation Script For Response Generator")
parser.add_argument("--base_model", required=True, type=str)
parser.add_argument("--name_or_path_for_fine_tuned_model", "-n", required=False, type=str, default="")
parser.add_argument("--prompt_type", required=False, type=str, default="")
parser.add_argument("--experiment_detail", required=True, type=str, default="")
parser.add_argument("--wandb_mode", required=False, type=str, default="online")

arguments = parser.parse_args()
arguments.fine_tuned_model = arguments.base_model if arguments.fine_tuned_model == "" else arguments.fine_tuned_model

# Initialize Wandb
wandb.init(project="emotion-chat-bot-ncu",
           group="Response Generator",
           job_type="fine-tuning",
           mode=arguments.wandb_mode,
           config={"base_model": arguments.base_model,
                   "name_or_path_for_fine_tuned_model": arguments.name_or_path_for_fine_tuned_model,
                   "experiment_detail": arguments.experiment_detail})

# Load Dataset
dataset = load_dataset("daily_dialog", split="train", num_proc=16, trust_remote_code=True)
dataset = dataset.remove_columns("act")
dataset = dataset.rename_column("emotion", "emotion_id")
emotion_labels: list = dataset.features["emotion_id"].feature.names
emotion_labels[0] = "neutral"
dataset = dataset.map(lambda samples: {
    "emotion": [[emotion_labels[emotion_id] for emotion_id in sample] for sample in samples]
}, input_columns="emotion_id", remove_columns="emotion_id", batched=True, num_proc=16)
dataset = dataset.map(lambda samples: {
    "dialog": [[dialog.strip() for dialog in sample] for sample in samples]
}, input_columns="dialog", batched=True, num_proc=16)
test_data = dataset.map(lambda samples: {
    "prompt": [[{"role": "user" if i % 2 == 0 else "assistant", "content": dialog}
                for i, dialog in enumerate(sample)] for sample in samples]
}, batched=True, num_proc=16)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(arguments.base_model, truncation=True, trust_remote_code=True)
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
                                 optim="paged_adamw_32bit", save_steps=1, logging_steps=1, learning_rate=2e-4,
                                 weight_decay=0.001, fp16=False, bf16=False, max_grad_norm=0.3, max_steps=-1,
                                 warmup_ratio=0.03, group_by_length=True, lr_scheduler_type="constant",
                                 report_to=["wandb"], gradient_checkpointing=True,
                                 gradient_checkpointing_kwargs={"use_reentrant": True})

# Load Model
base_model = AutoModelForCausalLM.from_pretrained(arguments.base_model, quantization_config=quantization_config,
                                                  device_map="auto" if torch.cuda.is_available() else "cpu",
                                                  low_cpu_mem_usage=True)

base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

# Setup Tuner
tuner = SFTTrainer(model=base_model, train_dataset=train_data, dataset_text_field="prompt", tokenizer=tokenizer,
                   peft_config=peft_parameters, args=train_params, max_seq_length=1024)

tuner.train()

tuner.model.save_pretrained(fine_tuned_model_name)

wandb.finish()
