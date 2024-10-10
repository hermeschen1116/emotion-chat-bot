from argparse import ArgumentParser
from dataclasses import Field, dataclass

import torch
from datasets import load_dataset
from transformers import HfArgumentParser
from transformers.hf_argparser import HfArg
from unsloth import FastLanguageModel

import wandb
from emotion_chat_bot.utils.CommonConfig import CommonScriptArguments, CommonWanDBArguments


@dataclass
class ScriptArguments(CommonScriptArguments):
	chat_template_file: Field[str] = HfArg(aliases="--chat-template-file", default="")


config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)

# Initialize Wandb
run = wandb.init(
	name=wandb_args.name,
	job_type=wandb_args.job_type,
	config=wandb_args.config,
	project=wandb_args.project,
	group=wandb_args.group,
	notes=wandb_args.notes,
	mode="offline",
	resume=wandb_args.resume,
)

# Load Dataset
dataset = load_dataset("Shotaro30678/daily_dialog_for_EP", split="test", num_proc=16, trust_remote_code=True)
dataset = dataset.take(100)

system_prompt: list = [{"role": "system", "content": {"emotion": "", "dialog": run.config["system_prompt"]}}]

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
### added
# tokenizer.pad_token_id = tokenizer.eos_token_id
# tokenizer.pad_token = tokenizer.eos_token
###

tokenizer.padding_side = "right"
tokenizer.clean_up_tokenization_spaces = True
# tokenizer.chat_template = run.config["chat_template"]
# tokenizer.add_special_tokens(run.config["special_tokens"])
base_model.resize_token_embeddings(len(tokenizer))


base_model = FastLanguageModel.get_peft_model(
	base_model,
	target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
	lora_alpha=run.config["lora_alpha"],
	lora_dropout=0.1,
	r=run.config["lora_rank"],
	bias="none",
	init_lora_weights=run.config["init_lora_weights"],
	modules_to_save=["lm_head", "embed_tokens"],
	use_rslora=True,
)

### added
# base_model.config.pad_token_id = tokenizer.pad_token_id
# base_model.config.use_cache = False
# base_model.config.pretraining_tp = 1
###

base_model.print_trainable_parameters()
FastLanguageModel.for_training(base_model)

batch_size = 1

all_outputs = []
pred = []

for i in range(0, len(dataset), batch_size):
	batch_sentences = dataset["text"][i : i + batch_size]

	inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)
	print(inputs["input_ids"].shape)

	# inputs = {k: v.to('cuda' if torch.cuda.is_available() else 'cpu') for k, v in inputs.items()}

	with torch.no_grad():
		outputs = base_model(**inputs)
		all_outputs.append(outputs["logits"])

final_outputs = torch.cat(all_outputs, dim=0)
pred = final_outputs.argmax(axis=1).cpu().numpy()
