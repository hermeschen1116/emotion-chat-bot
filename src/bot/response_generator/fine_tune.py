import tempfile
from argparse import ArgumentParser
from dataclasses import dataclass

import peft
import torch
import wandb
from datasets import load_from_disk, concatenate_datasets
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from transformers.hf_argparser import HfArg
from transformers.utils.hub import move_cache
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from libs.CommonConfig import CommonScriptArguments, CommonWanDBArguments, get_torch_device

move_cache()


@dataclass
class ScriptArguments(CommonScriptArguments):
    chat_template_file: str = HfArg(aliases="--chat-template-file", default="")
    pretraining_tp: int = HfArg(aliases="--pretraining-tp", default=None)
    enable_flash_attention: bool = HfArg(aliases="--enable-flash-attention", default=True)


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
dataset_path = run.use_artifact(wandb.config["dataset"]).download()
dataset = load_from_disk(dataset_path)
dataset = concatenate_datasets([dataset["train"], dataset["validation"]])

system_prompt: list = [{"role": "system", "content": {"emotion": "", "dialog": wandb.config["system_prompt"]}}]

dataset = dataset.map(lambda samples: {
    "prompt": [system_prompt + sample for sample in samples]
}, input_columns="prompt", batched=True, num_proc=16)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(wandb.config["base_model"])
tokenizer.padding_side = "right"
tokenizer.clean_up_tokenization_spaces = True
tokenizer.chat_template = wandb.config["chat_template"]
tokenizer.add_special_tokens(wandb.config["special_tokens"])

dataset = dataset.map(lambda samples: {
    "prompt": [tokenizer.apply_chat_template(sample, tokenize=False) for sample in samples]
}, input_columns="prompt", batched=True, num_proc=16)
wandb.config["example_prompt"] = dataset[0]["prompt"]


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

lora_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"]
)

# Load Model
base_model = AutoModelForCausalLM.from_pretrained(
    wandb.config["base_model"],
    quantization_config=quantization_config,
    use_flash_attention_2=args.enable_flash_attention,
    # attn_implementation="flash_attention_2",
    torch_dtype=torch.float16,
    use_cache=False,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
base_model.config.pretraining_tp = args.pretraining_tp
base_model.resize_token_embeddings(len(tokenizer))
base_model = peft.get_peft_model(base_model, lora_config)
base_model = base_model.merge_and_unload()

data_collator = DataCollatorForCompletionOnlyLM(
    wandb.config["response"],
    instruction_template=wandb.config["instruction"],
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
    dataloader_num_workers=16,
    optim=wandb.config["optim"],
    group_by_length=True,
    report_to=["wandb"],
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={
        "use_reentrant": True
    },
    auto_find_batch_size=True,
    torch_compile=False
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

with torch.autocast(get_torch_device()):
    tuner.train()

model_artifact = wandb.Artifact(
    wandb.config["fine_tuned_mode"],
    type="model",
    description=args.description_for_fine_tuned_model
)

tuner.model = torch.compile(tuner.model)
with tempfile.TemporaryDirectory() as temp_dir:
    tuner.save_model(temp_dir)
    model_artifact.add_dir(temp_dir)
    run.log_artifact(model_artifact)

wandb.finish()
