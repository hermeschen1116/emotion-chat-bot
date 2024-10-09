from argparse import ArgumentParser
from dataclasses import Field, dataclass

import wandb
from datasets import load_dataset
from peft.peft_model import PeftModel
from transformers import HfArgumentParser, TrainingArguments
from transformers.hf_argparser import HfArg
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from unsloth import FastLanguageModel

from emotion_chat_bot.utils.CommonConfig import CommonScriptArguments, CommonWanDBArguments


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
	name=wandb_args.name,
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
	"hermeschen1116/daily_dialog_for_RG",
	split="train+validation",
	num_proc=16,
	trust_remote_code=True,
)
# dataset = dataset.train_test_split(train_size=0.001)["train"]

system_prompt: list = [
	{
		"role": "system",
		"content": {"emotion": "", "dialog": run.config["system_prompt"]},
	}
]

dataset = dataset.map(
	lambda samples: {"prompt": [system_prompt + sample for sample in samples]},
	input_columns="prompt",
	batched=True,
	num_proc=16,
)

# Load Tokenizer
base_model, tokenizer = FastLanguageModel.from_pretrained(
	run.config["tokenizer"],
	attn_implementation="flash_attention_2",
	pretraining_tp=1,
	load_in_4bit=(run.config["init_lora_weights"] != "loftq"),
	use_cache=False,
	device_map="auto",
	low_cpu_mem_usage=True,
	trust_remote_code=True,
)
tokenizer.padding_side = "right"
tokenizer.clean_up_tokenization_spaces = True
tokenizer.chat_template = run.config["chat_template"]
tokenizer.add_special_tokens(run.config["special_tokens"])
base_model.resize_token_embeddings(len(tokenizer))

base_model = PeftModel.from_pretrained(base_model, run.use_model(run.config["base_model"]))
base_model.print_trainable_parameters()
FastLanguageModel.for_training(base_model)

dataset = dataset.map(
	lambda samples: {"prompt": [tokenizer.apply_chat_template(sample, tokenize=False) for sample in samples]},
	input_columns="prompt",
	batched=True,
	num_proc=16,
)
run.config["example_prompt"] = dataset[0]["prompt"]

special_tokens_map: dict = dict(zip(tokenizer.all_special_tokens, [[ids] for ids in tokenizer.all_special_ids]))
data_collator = DataCollatorForCompletionOnlyLM(
	special_tokens_map[run.config["response_template"]],
	instruction_template=special_tokens_map[run.config["instruction_template"]],
	tokenizer=tokenizer,
)

trainer_arguments = TrainingArguments(
	output_dir="./checkpoints",
	overwrite_output_dir=True,
	per_device_train_batch_size=4,
	gradient_accumulation_steps=1,
	learning_rate=run.config["learning_rate"],
	weight_decay=run.config["weight_decay"],
	max_grad_norm=run.config["max_grad_norm"],
	num_train_epochs=run.config["num_epochs"],
	lr_scheduler_type="constant",
	warmup_ratio=run.config["warmup_ratio"],
	max_steps=run.config["max_steps"],
	logging_steps=25,
	save_steps=25,
	save_total_limit=5,
	bf16=True,
	fp16=False,
	dataloader_num_workers=12,
	optim=run.config["optim"],
	group_by_length=True,
	report_to=["wandb"],
	push_to_hub=True,
	hub_model_id="response_generator_for_emotion_chat_bot",
	gradient_checkpointing=True,
	gradient_checkpointing_kwargs={"use_reentrant": True},
	auto_find_batch_size=True,
	torch_compile=False,
	include_tokens_per_second=True,
	include_num_input_tokens_seen=True,
	neftune_noise_alpha=run.config["neftune_noise_alpha"],
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
	dataset_num_proc=16,
)

tuner.train()

wandb.finish()
