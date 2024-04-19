import os
from dataclasses import dataclass

import huggingface_hub
import torch
import wandb
from datasets import load_from_disk
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser
from transformers.utils.hub import move_cache
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from libs.Config import WanDBArguments, TrainerArguments

move_cache()

# commandline input


@dataclass
class ScriptArguments:
    huggingface_api_token: str = ""
    wandb_api_token: str = ""
    chat_template_file: str = None
    pretraining_tp: int = None
    name_for_fine_tuned_model: str = ""
    description_for_fine_tuned_model: str = ""
    path_for_fine_tuned_model: str = "./model"


parser = HfArgumentParser((ScriptArguments, WanDBArguments, LoraConfig, BitsAndBytesConfig, TrainerArguments))
args, wandb_args, lora_config, quantization_config, trainer_arguments = parser.parse_args()

chat_template: dict = eval(open(args.chat_template_file, "r", encoding="utf-8", closefd=True).read())
# parser.add_argument("--base_model", required=True, type=str)
# parser.add_argument("--tokenizer", required=False, type=str, default=None)
# parser.add_argument("--name_or_path_for_fine_tuned_model", "-n", required=False, type=str, default="")
# parser.add_argument("--num_epochs", required=False, type=int, default=1)
# parser.add_argument("--system_prompt", required=False, type=str, default=None)

load_dotenv(encoding="utf-8")
huggingface_hub.login(token=os.environ.get("HF_TOKEN", args.huggingface_api_token), add_to_git_credential=True)
wandb.login(key=os.environ.get("WANDB_API_KEY", args.wandb_api_token), relogin=True)

# Initialize Wandb
run = wandb.init(wandb_args)
wandb.config["chat_template"] = chat_template["template"]
wandb.config["instruction_template"] = chat_template["instruction"]
wandb.config["response_template"] = chat_template["response"]
wandb.config["special_tokens"] = chat_template["special_tokens"]
# run = wandb.init(
#     job_type="fine-tuning",
#     config=wandb_config,
#     project="emotion-chat-bot-ncu",
#     group="Response Generator",
#     notes=arguments.experiment_detail,
#     mode=arguments.wandb_mode,
#     resume="auto",
#     allow_val_change=True
# )

# Load Dataset
dataset_path = run.use_artifact("daily_dialog_for_RG_train:latest").download()
dataset = load_from_disk(dataset_path)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(wandb_args.config["base_model"])
tokenizer.padding_side = "right"
tokenizer.clean_up_tokenization_spaces = True
tokenizer.chat_template = wandb.config["template"]
tokenizer.add_special_tokens(wandb.config["special_tokens"], replace_additional_special_tokens=True)


dataset = dataset.map(lambda sample: {
    "prompt": tokenizer.apply_chat_template(sample,
                                            tokenize=False,
                                            padding=True,
                                            max_length=4096,
                                            return_tensors="pt"
                                            )
}, input_columns="prompt", num_proc=16)
wandb.config["example_prompt"] = dataset[0]["prompt"]

# Configurations
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=False
# )
quantization_config = quantization_config if torch.cuda.is_available() else None

# lora_config = LoraConfig(
#     lora_alpha=16,
#     lora_dropout=0.1,
#     r=8,
#     bias="none",
#     task_type="CAUSAL_LM"
# )

# trainer_arguments = TrainingArguments(
#     output_dir="./checkpoints",
#     overwrite_output_dir=True,
#     per_device_train_batch_size=4,
#     gradient_accumulation_steps=1,
#     learning_rate=2e-4,
#     weight_decay=0.001,
#     max_grad_norm=0.3,
#     num_train_epochs=wandb.config["num_epochs"],
#     lr_scheduler_type="constant",
#     warmup_ratio=0.03,
#     max_steps=-1,
#     logging_steps=25,
#     save_steps=25,
#     save_total_limit=5,
#     bf16=False,
#     fp16=False,
#     dataloader_num_workers=16,
#     optim="paged_adamw_32bit",
#     group_by_length=True,
#     report_to=["wandb"],
#     gradient_checkpointing=True,
#     gradient_checkpointing_kwargs={"use_reentrant": True},
#     auto_find_batch_size=True,
#     torch_compile=False,
#     resume_from_checkpoint=True
# )

# Load Model
base_model = AutoModelForCausalLM.from_pretrained(
    wandb_args.config["base_model"],
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2",
    pretraining_tp=args.pretraining_tp,
    use_cache=False,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
base_model.resize_token_embeddings(len(tokenizer))
wandb.config["base_model_configuration"] = base_model.config.to_dict()

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

tuner.model = torch.compile(tuner.model)
tuner.save_model(args.path_for_fine_tuned_model)
model_artifact = wandb.Artifact(
    args.name_for_fine_tuned_model,
    type="model",
    description=args.description_for_fine_tuned_model,
    incremental=True
)
model_artifact.add_dir(args.path_for_fine_tuned_model)
run.log_artifact(model_artifact)

wandb.finish()
