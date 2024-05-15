import tempfile
from argparse import ArgumentParser
from dataclasses import dataclass

import torch
import wandb
from datasets import load_from_disk, concatenate_datasets
from peft import PeftModel
from transformers import HfArgumentParser, TrainingArguments
from transformers.hf_argparser import HfArg
from transformers.utils.hub import move_cache
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from unsloth import FastLanguageModel

from libs import CommonScriptArguments, CommonWanDBArguments

move_cache()


@dataclass
class ScriptArguments(CommonScriptArguments):
    chat_template_file: str = HfArg(aliases="--chat-template-file", default="")


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
dataset_path = run.use_artifact(wandb.config["dataset"]).download()
dataset = load_from_disk(dataset_path)
dataset = concatenate_datasets([dataset["train"], dataset["validation"]])
# dataset = dataset.train_test_split(train_size=0.001)["train"]

system_prompt: list = [{"role": "system", "content": {"emotion": "", "dialog": wandb.config["system_prompt"]}}]

dataset = dataset.map(lambda samples: {
    "prompt": [system_prompt + sample for sample in samples]
}, input_columns="prompt", batched=True, num_proc=16)

# Load Tokenizer
base_model, tokenizer = FastLanguageModel.from_pretrained(
    wandb.config["tokenizer"],
    attn_implementation="flash_attention_2",
    pretraining_tp=1,
    load_in_4bit=(wandb.config["init_lora_weights"] != "loftq"),
    use_cache=False,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
tokenizer.padding_side = "right"
tokenizer.clean_up_tokenization_spaces = True
tokenizer.chat_template = wandb.config["chat_template"]
tokenizer.add_special_tokens(wandb.config["special_tokens"])
base_model.resize_token_embeddings(len(tokenizer))

base_model = PeftModel.from_pretrained(base_model, run.use_model(wandb.config["base_model"]))
base_model.print_trainable_parameters()
FastLanguageModel.for_training(base_model)

dataset = dataset.map(lambda samples: {
    "prompt": [tokenizer.apply_chat_template(sample, tokenize=False) for sample in samples]
}, input_columns="prompt", batched=True, num_proc=16)
wandb.config["example_prompt"] = dataset[0]["prompt"]

special_tokens_map: dict = dict(zip(tokenizer.all_special_tokens, [[ids] for ids in tokenizer.all_special_ids]))
data_collator = DataCollatorForCompletionOnlyLM(
    special_tokens_map[wandb.config["response_template"]],
    instruction_template=special_tokens_map[wandb.config["instruction_template"]],
    tokenizer=tokenizer
)

trainer_arguments = TrainingArguments(
    output_dir="./checkpoints",
    overwrite_output_dir=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    learning_rate=wandb.config["learning_rate"],
    weight_decay=wandb.config["weight_decay"],
    max_grad_norm=wandb.config["max_grad_norm"],
    num_train_epochs=wandb.config["num_epochs"],
    lr_scheduler_type="constant",
    warmup_ratio=wandb.config["warmup_ratio"],
    max_steps=wandb.config["max_steps"],
    logging_steps=25,
    save_steps=25,
    save_total_limit=5,
    bf16=True,
    fp16=False,
    dataloader_num_workers=12,
    optim=wandb.config["optim"],
    group_by_length=True,
    report_to=["wandb"],
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": True
    },
    auto_find_batch_size=True,
    torch_compile=False,
    neftune_noise_alpha=wandb.config["neftune_noise_alpha"]
)

# Setup Tuner
tuner = SFTTrainer(
    model=base_model,
    args=trainer_arguments,
    data_collator=data_collator,
    train_dataset=dataset,
    dataset_text_field="prompt",
    tokenizer=tokenizer,
    max_seq_length=4096,
    dataset_num_proc=16
)

tuner.train()

model_artifact = wandb.Artifact(
    wandb.config["fine_tuned_model"],
    type="model"
)

tuner.model = torch.compile(tuner.model)
with tempfile.TemporaryDirectory() as temp_dir:
    tuner.model.save_pretrained(temp_dir, save_embedding_layers=True)
    model_artifact.add_dir(temp_dir)
    run.log_artifact(model_artifact)

wandb.finish()
