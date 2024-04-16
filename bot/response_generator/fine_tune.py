import argparse
import os

import huggingface_hub
import torch
from accelerate import DataLoaderConfiguration

import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from transformers.utils.hub import move_cache
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# commandline inputs
parser = argparse.ArgumentParser(prog="Fine-Tuning", description="Fine-Tuning Script For Response Generator")
parser.add_argument("--base_model", required=True, type=str)
parser.add_argument("--tokenizer", required=False, type=str, default=None)
parser.add_argument("--name_or_path_for_fine_tuned_model", "-n", required=False, type=str, default="")
parser.add_argument("--experiment_detail", required=True, type=str)
parser.add_argument(
    "--wandb_mode",
    required=False,
    type=str,
    choices=["online", "offline", "disabled"],
    default="online"
)
parser.add_argument("--num_epochs", required=False, type=int, default=1)
parser.add_argument("--enable_flash_attention_2", required=False, type=bool, default=True)
parser.add_argument(
    "--system_prompt_mode",
    required=False,
    type=str,
    choices=["default", "custom", "disabled"],
    default="disabled"
)
parser.add_argument("--system_prompt", required=False, type=str, default=None)
parser.add_argument("--chat_template_file", required=True, type=str, default="")

arguments = parser.parse_args()
arguments.tokenizer = arguments.base_model if arguments.tokenizer is None else arguments.tokenizer
if arguments.system_prompt_mode == "disabled":
    arguments.system_prompt = None
else:
    arguments.system_prompt = "" if arguments.system_prompt_mode == "default" else arguments.system_prompt
chat_template: dict = eval(open(arguments.chat_template_file, "r", encoding="utf-8", closefd=True).read())

load_dotenv(encoding="utf-8")
huggingface_hub.login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=True)
wandb.login(key=os.environ.get("WANDB_API_KEY", ""), relogin=True)

# Initialize Wandb
wandb_config: dict = {
    "base_model": arguments.base_model,
    "tokenizer": arguments.tokenizer,
    "name_or_path_for_fine_tuned_model": arguments.name_or_path_for_fine_tuned_model,
    "system_prompt": arguments.system_prompt,
    "chat_template": chat_template["template"],
    "instruction_template": chat_template["instruction"],
    "response_template": chat_template["response"],
    "special_tokens": chat_template["special_tokens"]
}
wandb.init(
    job_type="fine-tuning",
    config=wandb_config,
    project="emotion-chat-bot-ncu",
    group="Response Generator",
    notes=arguments.experiment_detail,
    mode=arguments.wandb_mode,
    resume="auto",
    allow_val_change=True
)

# Load Dataset
dataset = load_dataset("daily_dialog",
                       split="train",
                       num_proc=16,
                       trust_remote_code=True).remove_columns("act")
dataset = dataset.rename_column("emotion", "emotion_id")
emotion_labels: list = dataset.features["emotion_id"].feature.names
emotion_labels[0] = "neutral"
dataset = dataset.map(lambda samples: {
    "emotion": [[emotion_labels[emotion_id] for emotion_id in sample] for sample in samples]
}, input_columns="emotion_id", remove_columns="emotion_id", batched=True, num_proc=16)
dataset = dataset.map(lambda samples: {
    "dialog": [[dialog.strip() for dialog in sample] for sample in samples]
}, input_columns="dialog", batched=True, num_proc=16)
dataset = dataset.map(lambda samples: {
    "prompt": [[{
        "role": "user" if i % 2 == 0 else "assistant",
        "content": {"emotion": emotion, "dialog": dialog}
    }
        for i, (emotion, dialog) in enumerate(zip(sample[0], sample[1]))]
        for sample in zip(samples["emotion"], samples["dialog"])]
}, remove_columns=["emotion", "dialog"], batched=True, num_proc=16)
dataset = dataset.map(lambda samples: {
    "prompt": [sample[:-1] if len(sample) % 2 == 1 else sample for sample in samples]
}, input_columns="prompt", batched=True, num_proc=16)
if arguments.system_prompt_mode != "disabled":
    dataset = dataset.map(lambda samples: {
        "prompt": [[{
            "role": "system",
            "content": {"emotion": None, "dialog": arguments.system_prompt}
        }] + sample for sample in samples]
    }, input_columns="prompt", batched=True, num_proc=16)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(arguments.base_model)
tokenizer.padding_side = "right"
tokenizer.clean_up_tokenization_spaces = True
tokenizer.chat_template = chat_template["template"]
tokenizer.add_special_tokens(chat_template["special_tokens"], replace_additional_special_tokens=True)


def prompt_compose(sample: str) -> str:
    return tokenizer.apply_chat_template(sample,
                                         tokenize=False,
                                         padding=True,
                                         max_length=4096,
                                         return_tensors="pt"
                                         )


dataset = dataset.map(lambda sample: {
    "prompt": prompt_compose(sample)
}, input_columns="prompt", num_proc=16)
wandb.config["example_prompt"] = dataset[0]["prompt"]
# dataset = dataset.train_test_split(test_size=0.1)

# Configurations
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)
quantization_config = quantization_config if torch.cuda.is_available() else None
wandb.config["quantization_configuration"] = quantization_config.to_dict() if quantization_config is not None else {}

lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)
wandb.config["lora_configuration"] = lora_config.to_dict()

# dataloader_config = DataLoaderConfiguration(
#     dispatch_batches=None,
#     split_batches=False,
#     even_batches=True,
#     use_seedable_sampler=True
# )

trainer_arguments = TrainingArguments(
    output_dir="./checkpoints",
    overwrite_output_dir=True,
    # evaluation_strategy="steps",
    per_device_train_batch_size=4,
    # per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    # eval_accumulation_steps=1,
    # eval_delay=0.5,
    learning_rate=2e-4,
    weight_decay=0.001,
    max_grad_norm=0.3,
    num_train_epochs=arguments.num_epochs,
    lr_scheduler_type="constant",
    warmup_ratio=0.03,
    max_steps=-1,
    logging_steps=25,
    save_steps=25,
    save_total_limit=5,
    bf16=False,
    fp16=False,
    dataloader_num_workers=16,
    # load_best_model_at_end=True,
    # metric_for_best_model="loss",
    optim="paged_adamw_32bit",
    group_by_length=True,
    report_to=["wandb"],
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    auto_find_batch_size=True,
    torch_compile=False,
    resume_from_checkpoint=True
)
wandb.config["trainer_arguments"] = trainer_arguments.to_dict()

flash_attention: str = "flash_attention_2" if arguments.enable_flash_attention_2 else None
# Load Model
base_model = AutoModelForCausalLM.from_pretrained(
    arguments.base_model,
    quantization_config=quantization_config,
    attn_implementation=flash_attention,
    pretraining_tp=1,
    use_cache=False,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
base_model.resize_token_embeddings(len(tokenizer))
wandb.config["base_model_configuration"] = base_model.config.to_dict()

data_collator = DataCollatorForCompletionOnlyLM(
    chat_template["response"],
    instruction_template=chat_template["instruction"],
    tokenizer=tokenizer
)

# Setup Tuner
tuner = SFTTrainer(
    model=base_model,
    args=trainer_arguments,
    data_collator=data_collator,
    train_dataset=dataset,
    # eval_dataset=dataset["test"],
    peft_config=lora_config,
    dataset_text_field="prompt",
    tokenizer=tokenizer,
    max_seq_length=4096,
    dataset_num_proc=16
)

tuner.train()

tuner.model = torch.compile(tuner.model)
tuner.save_model("./model")
wandb.save("./model")

wandb.finish()
