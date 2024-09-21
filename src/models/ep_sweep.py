from argparse import ArgumentParser

import numpy as np
import torch
from datasets import load_dataset
from libs.CommonConfig import CommonScriptArguments, CommonWanDBArguments
from libs.CommonUtils import login_to_service
from libs.DataProcess import throw_out_partial_row_with_a_label
from peft import LoraConfig, get_peft_model
from sklearn.utils.class_weight import compute_class_weight
from torch import Tensor, nn
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, Trainer, TrainingArguments

import wandb

login_to_service()

config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((CommonScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)

# Define sweep configuration
sweep_configuration = {
	"method": "random",
	"name": "sweep",
	"metric": {"goal": "maximize", "name": "Accuracy"},
	"parameters": {
		"batch_size": {"values": [32, 64]},
		"num_train_epochs": {"values": [8]},
		"learning_rate": {"max": 0.1, "min": 0.0001},
		"lora_alpha": {"values": [16, 32, 64]},
		"lora_dropout": {"values": [0.1, 0.2, 0.3]},
		"lora_rank": {"values": [16, 32, 64]},
		"init_lora_weights": {"values": [True, False]},
		"use_rslora": {"values": [True, False]},
		"focal_gamma": {"values": [0, 1, 2]},
		"weight_decay": {"max": 0.3, "min": 0.0},
		"warmup_ratio": {"max": 0.1, "min": 0.0},
		"max_steps": {"max": 5000, "min": -1},
		"max_grad_norm": {"max": 1.0, "min": 0.1},
	},
}

# Initialize sweep
sweep_id = wandb.sweep(sweep=sweep_configuration, project="emotion-chat-bot-ncu-ep-sweep")


def main():
	# Initialize wandb run
	run = wandb.init(
		name=wandb_args.name,
		job_type=wandb_args.job_type,
		config=wandb_args.config,
		project=wandb_args.project,
		group=wandb_args.group,
		notes=wandb_args.notes,
	)

	# Fetch hyperparameters from `wandb.config`
	# trainer args
	batch_size = wandb.config.batch_size
	num_train_epochs = wandb.config.num_train_epochs
	learning_rate = wandb.config.learning_rate
	weight_decay = wandb.config.weight_decay
	warmup_ratio = wandb.config.warmup_ratio
	max_steps = wandb.config.max_steps
	max_grad_norm = wandb.config.max_grad_norm

	# lora args
	lora_alpha = wandb.config.lora_alpha
	lora_dropout = wandb.config.lora_dropout
	lora_rank = wandb.config.lora_rank
	init_lora_weights = wandb.config.init_lora_weights
	use_rslora = wandb.config.use_rslora
	# focal loss args
	focal_gamma = wandb.config.focal_gamma

	# Load dataset
	dataset = load_dataset(
		run.config["dataset"],
		num_proc=16,
		trust_remote_code=True,
	)
	emotion_labels: list = dataset["train"].features["label"].names
	num_emotion_labels: int = len(emotion_labels)

	train_dataset = throw_out_partial_row_with_a_label(dataset["train"], run.config["neutral_keep_ratio"], 0)
	train_dataset = train_dataset.take(8192)
	validation_dataset = dataset["validation"]
	validation_dataset = validation_dataset.take(1024)

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
		device_map="cuda",
		low_cpu_mem_usage=True,
		trust_remote_code=True,
	)

	peft_config = LoraConfig(
		task_type="SEQ_CLS",
		lora_alpha=lora_alpha,
		lora_dropout=lora_dropout,
		r=lora_rank,
		bias="none",
		init_lora_weights=init_lora_weights,
		use_rslora=use_rslora,
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
		sentiment_pred: Tensor = torch.tensor(
			[[label] for label in prediction.predictions.argmax(-1).tolist()]
		).flatten()

		accuracy = multiclass_accuracy(sentiment_true, sentiment_pred, num_classes=num_emotion_labels)
		f1 = multiclass_f1_score(
			sentiment_true,
			sentiment_pred,
			num_classes=num_emotion_labels,
			average="weighted",
		)

		wandb.log(
			{
				"Accuracy": accuracy,
				"F1-score": f1,
			}
		)

		return {
			"Accuracy": accuracy,
			"F1-score": f1,
		}

	y = train_dataset["label"].tolist()
	class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)

	class FocalLoss(nn.Module):
		def __init__(self, alpha=None, gamma=2, ignore_index=-100, reduction="mean"):
			super().__init__()
			# use standard CE loss without reduction as basis
			self.CE = nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)
			self.alpha = alpha
			self.gamma = gamma
			self.reduction = reduction

		def forward(self, input, target):
			"""
			input (B, N)
			target (B)
			"""
			minus_logpt = self.CE(input, target)
			pt = torch.exp(-minus_logpt)  # don't forget the minus here
			focal_loss = (1 - pt) ** self.gamma * minus_logpt

			# apply class weights
			if self.alpha is not None:
				focal_loss *= self.alpha.gather(0, target)

			if self.reduction == "mean":
				focal_loss = focal_loss.mean()
			elif self.reduction == "sum":
				focal_loss = focal_loss.sum()
			return focal_loss

	class_weights = torch.tensor(class_weights, dtype=torch.float).to("cuda")
	loss_fct = FocalLoss(alpha=class_weights, gamma=focal_gamma)

	class CustomTrainer(Trainer):
		def compute_loss(self, model, inputs, return_outputs=False):
			labels = inputs.get("labels")
			outputs = model(**inputs)
			logits = outputs.get("logits")
			loss = loss_fct(logits, labels)
			return (loss, outputs) if return_outputs else loss

	logging_steps = len(dataset["train"]) // batch_size
	trainer_arguments = TrainingArguments(
		output_dir="./checkpoints",
		overwrite_output_dir=True,
		per_device_train_batch_size=batch_size,
		per_device_eval_batch_size=batch_size,
		gradient_accumulation_steps=1,
		learning_rate=learning_rate,
		lr_scheduler_type="constant",
		weight_decay=weight_decay,
		max_grad_norm=max_grad_norm,
		num_train_epochs=num_train_epochs,
		warmup_ratio=warmup_ratio,
		max_steps=max_steps,
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

	tuner = CustomTrainer(
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

	if hasattr(tuner.model, "config"):
		tuner.model.config.save_pretrained("model_test")
	tuner.save_model("model_test")

	wandb.finish()


# Start sweep job
wandb.agent(sweep_id, function=main, count=4)
