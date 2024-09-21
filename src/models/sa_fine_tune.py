from argparse import ArgumentParser

import torch
from datasets import load_dataset
from libs.CommonConfig import CommonScriptArguments, CommonWanDBArguments
from libs.DataProcess import throw_out_partial_row_with_a_label
from peft import LoraConfig, get_peft_model
from torch import Tensor
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, Trainer, TrainingArguments

import wandb

config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((CommonScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)

run = wandb.init(
	name=wandb_args.name,
	job_type=wandb_args.job_type,
	config=wandb_args.config,
	project=wandb_args.project,
	group=wandb_args.group,
	notes=wandb_args.notes,
)

dataset = load_dataset(
	run.config["dataset"],
	num_proc=16,
	trust_remote_code=True,
)
emotion_labels: list = dataset["train"].features["label"].names
num_emotion_labels: int = len(emotion_labels)

train_dataset = throw_out_partial_row_with_a_label(dataset["train"], run.config["neutral_keep_ratio"], 0)
validation_dataset = dataset["validation"]

tokenizer = AutoTokenizer.from_pretrained(
	run.config["base_model"],
	padding_side="right",
	clean_up_tokenization_spaces=True,
	trust_remote_code=True,
)

base_model = AutoModelForSequenceClassification.from_pretrained(
	run.config["base_model"],
	num_labels=num_emotion_labels,
	id2label={k: v for k, v in enumerate(emotion_labels)},
	label2id={v: k for k, v in enumerate(emotion_labels)},
	use_cache=False,
	device_map="auto",
	low_cpu_mem_usage=True,
	trust_remote_code=True,
)

peft_config = LoraConfig(
	task_type="SEQ_CLS",
	lora_alpha=run.config["lora_alpha"],
	lora_dropout=run.config["lora_dropout"],
	r=run.config["lora_rank"],
	bias="none",
	init_lora_weights=run.config["init_lora_weights"],
	use_rslora=run.config["use_rslora"],
)
base_model = get_peft_model(base_model, peft_config)

train_dataset = train_dataset.map(
	lambda samples: {
		"input_ids": [tokenizer.encode(sample, padding="max_length", truncation=True) for sample in samples],
	},
	input_columns=["text"],
	batched=True,
	num_proc=16,
)
train_dataset.set_format("torch")
validation_dataset = validation_dataset.map(
	lambda samples: {
		"input_ids": [tokenizer.encode(sample, padding="max_length", truncation=True) for sample in samples],
	},
	input_columns=["text"],
	batched=True,
	num_proc=16,
)
validation_dataset.set_format("torch")


def compute_metrics(prediction) -> dict:
	sentiment_true: Tensor = torch.tensor([[label] for label in prediction.label_ids.tolist()]).flatten()
	sentiment_pred: Tensor = torch.tensor([[label] for label in prediction.predictions.argmax(-1).tolist()]).flatten()

	return {
		"Accuracy": multiclass_accuracy(sentiment_true, sentiment_pred, num_classes=num_emotion_labels),
		"F1-score": multiclass_f1_score(
			sentiment_true,
			sentiment_pred,
			num_classes=num_emotion_labels,
			average="weighted",
		),
	}


per_device_batch_size: int = 8
logging_steps: int = len(dataset["train"]) // per_device_batch_size
trainer_arguments = TrainingArguments(
	output_dir="./checkpoints",
	overwrite_output_dir=True,
	per_device_train_batch_size=8,
	per_device_eval_batch_size=8,
	gradient_accumulation_steps=1,
	learning_rate=run.config["learning_rate"],
	lr_scheduler_type="constant",
	weight_decay=run.config["weight_decay"],
	max_grad_norm=run.config["max_grad_norm"],
	num_train_epochs=run.config["num_train_epochs"],
	warmup_ratio=run.config["warmup_ratio"],
	max_steps=run.config["max_steps"],
	logging_steps=logging_steps,
	log_level="error",
	save_steps=500,
	save_total_limit=2,
	save_strategy="epoch",
	eval_strategy="epoch",
	load_best_model_at_end=True,
	fp16=True,
	bf16=False,
	dataloader_num_workers=12,
	optim=run.config["optim"],
	group_by_length=True,
	report_to=["wandb"],
	hub_model_id=run.config["fine_tuned_model"],
	gradient_checkpointing=True,
	gradient_checkpointing_kwargs={"use_reentrant": True},
	auto_find_batch_size=True,
	torch_compile=False,
	include_tokens_per_second=True,
	include_num_input_tokens_seen=True,
)

tuner = Trainer(
	model=base_model,
	args=trainer_arguments,
	compute_metrics=compute_metrics,
	train_dataset=train_dataset,
	eval_dataset=validation_dataset,
	tokenizer=tokenizer,
)

tuner.train()

tuner.model = torch.compile(tuner.model)
tuner.model = tuner.model.merge_and_unload(progressbar=True)
tuner.push_to_hub()

wandb.finish()
