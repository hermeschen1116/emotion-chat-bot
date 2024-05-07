import tempfile
from dataclasses import dataclass
from typing import Optional

import torch
import wandb
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments
from transformers.hf_argparser import HfArg
from transformers.utils.hub import move_cache
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from libs.CommonConfig import CommonScriptArguments, CommonWanDBArguments

move_cache()

# commandline input


@dataclass
class ScriptArguments(CommonScriptArguments):
    chat_template_file: str = HfArg(aliases="--chat-template-file", default="")
    fine_tuned_model_name: Optional[str] = HfArg(aliases="--fine-tuned-model-name", default="response_generator")


parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments, LoraConfig, TrainingArguments))
args, wandb_args, lora_config, trainer_arguments = parser.parse_args()

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

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(wandb.config["base_model"])
tokenizer.padding_side = "right"
tokenizer.clean_up_tokenization_spaces = True
tokenizer.chat_template = wandb.config["template"]
tokenizer.add_special_tokens(wandb.config["special_tokens"], replace_additional_special_tokens=True)

wandb.config["example_prompt"] = tokenizer.apply_chat_template(dataset[0]["prompt"], tokenize=False)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False
)

# Load Model
base_model = AutoModelForCausalLM.from_pretrained(
    wandb.config["base_model"],
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2",
    pretraining_tp=1,
    use_cache=False,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
base_model.resize_token_embeddings(len(tokenizer))

data_collator = DataCollatorForCompletionOnlyLM(
    wandb.config["response"],
    instruction_template=wandb.config["instruction"],
    tokenizer=tokenizer
)

# Setup Tuner
tuner = SFTTrainer(
    model=base_model,
    args=trainer_arguments,
    data_collator=data_collator,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="prompt",
    tokenizer=tokenizer,
    max_seq_length=4096,
    dataset_num_proc=16
)

tuner.train()

model_artifact = wandb.Artifact(
    args.fine_tuned_model_name,
    type="model",
    description=args.description_for_fine_tuned_model
)

tuner.model = torch.compile(tuner.model)
with tempfile.TemporaryDirectory() as temp_dir:
    tuner.save_model(temp_dir)
    model_artifact.add_dir(temp_dir)
    run.log_artifact(model_artifact)

wandb.finish()
