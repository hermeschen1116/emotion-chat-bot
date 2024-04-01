import os

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

load_dotenv(encoding="utf-8")
login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=True)

# Load Dataset
dataset = load_dataset("daily_dialog", split="train", num_proc=8, trust_remote_code=True)

# Process Dataset
dataset = dataset.remove_columns("act")

# Process Emotion Feature (ID to Label)
dataset = dataset.rename_column("emotion", "emotion_id")
emotion_labels: list = dataset.features["emotion_id"].feature.names
emotion_labels[0] = "neutral"
dataset = dataset.map(
    lambda samples: {"emotion": [[emotion_labels[emotion_id] for emotion_id in sample] for sample in samples]},
    input_columns="emotion_id", remove_columns="emotion_id", batched=True, num_proc=8)
# Add Agent Feature
agents: dict = {0: 'user', 1: 'bot'}
dataset = dataset.map(
    lambda samples: {"agent": [[agents[i % 2] for i in range(len(sample))] for sample in samples["dialog"]]},
    batched=True, num_proc=8)
# Remove Useless Spaces in Dialog
dataset = dataset.map(lambda samples: {"dialog": [[dialog.strip() for dialog in sample] for sample in samples]},
                      input_columns="dialog", batched=True, num_proc=8)

# Script Format ( {agent}({emotion}): {dialog} )
# prompt_type: str = "script"
# train_data = dataset.map(lambda samples: {
#     "lines": [[f"{agent}({emotion}): {dialog}" for agent, emotion, dialog in zip(sample[0], sample[1], sample[2])] for
#               sample in zip(samples["agent"], samples["emotion"], samples["dialog"])]}, batched=True, num_proc=8)
enter = "\n"  # for Python 3.11
# train_data = train_data.map(
#     lambda samples: {"prompt": [f"""### {f'{enter}### '.join(sample)}""" for sample in samples]}, input_columns="lines",
#     remove_columns=["agent", "emotion", "dialog", "lines"], batched=True, num_proc=8)

# JSON format
# prompt_type = "JSON"
#
# train_data = dataset.map(lambda samples: {"lines": [
#     [f'{{"agent": "{agent}", "emotion": "{emotion}", "dialog": "{dialog}"}}' for agent, emotion, dialog in
#      zip(sample[0], sample[1], sample[2])] for sample in zip(samples["agent"], samples["emotion"], samples["dialog"])]},
#                          batched=True, num_proc=8)
#
# train_data = train_data.map(lambda samples: {"prompt": [f"[{f',{enter}'.join(sample)}]" for sample in samples]},
#                             input_columns="lines", remove_columns=["agent", "emotion", "dialog", "lines"], batched=True,
#                             num_proc=8)

# History format
prompt_type = "history"

train_data = dataset.map(lambda samples: {"history": ["\n".join(sample[:-1]) for sample in samples]},
                         input_columns="dialog", batched=True, num_proc=8)

train_data = train_data.map(lambda samples: {
    "prompt": [f"HISTORY: {sample[0]}{enter}EMOTION: {sample[1][-1]}{enter}DIALOG: {sample[2][-1]}" for sample in
               zip(samples["history"], samples["emotion"], samples["dialog"])]},
                            remove_columns=["agent", "emotion", "dialog", "history"], batched=True, num_proc=8)

# Fine Tune
base_model_name: str = "meta-llama/Llama-2-7b-hf"
fine_tuned_model_name: str = f"response_generator_non_chat_llama_on_daily_dialog_type_{prompt_type}"

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
