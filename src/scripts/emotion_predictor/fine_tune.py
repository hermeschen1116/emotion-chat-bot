from argparse import ArgumentParser

import numpy as np
import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch import Tensor, nn
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, HfArgumentParser, Trainer, TrainingArguments

from emotion_chat_bot.utils.CommonConfig import CommonScriptArguments, CommonWanDBArguments
from emotion_chat_bot.utils.CommonUtils import login_to_service
from emotion_chat_bot.utils.DataProcess import throw_out_partial_row_with_a_label

login_to_service()

config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((CommonScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)

run = wandb.init(config=wandb_args.config, project=wandb_args.project, group=wandb_args.group)

dataset = load_dataset(run.config["dataset"], num_proc=16, trust_remote_code=True)
emotion_labels: list = dataset["train"].features["label"].names
num_emotion_labels: int = len(emotion_labels)

train_dataset = throw_out_partial_row_with_a_label(dataset["train"], run.config["neutral_keep_ratio"], 0)
validation_dataset = dataset["validation"]

tokenizer = AutoTokenizer.from_pretrained(
	run.config["base_model"], padding_side="right", clean_up_tokenization_spaces=True, trust_remote_code=True
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
		"input_ids": [tokenizer.encode(sample, padding="max_length", truncation=True) for sample in samples]
	},
	input_columns=["text"],
	batched=True,
	num_proc=16,
)
train_dataset.set_format("torch")
validation_dataset = validation_dataset.map(
	lambda samples: {
		"input_ids": [tokenizer.encode(sample, padding="max_length", truncation=True) for sample in samples]
	},
	input_columns=["text"],
	batched=True,
	num_proc=16,
)
validation_dataset.set_format("torch")


def compute_metrics(prediction) -> dict:
	sentiment_true: Tensor = torch.tensor([[label] for label in prediction.label_ids.tolist()]).flatten()
	sentiment_pred: Tensor = torch.tensor([[label] for label in prediction.predictions.argmax(-1).tolist()]).flatten()
	balanced_acc = balanced_accuracy_score(sentiment_true, sentiment_pred)

	accuracy = multiclass_accuracy(sentiment_true, sentiment_pred, num_classes=num_emotion_labels)
	f1_weighted = multiclass_f1_score(
		sentiment_true, sentiment_pred, num_classes=num_emotion_labels, average="weighted"
	)

	f1_per_class = multiclass_f1_score(sentiment_true, sentiment_pred, num_classes=num_emotion_labels, average=None).to(
		"cuda"
	)

	weighted_f1_per_class = f1_per_class * class_weights
	non_zero_count = (weighted_f1_per_class != 0).sum()
	weighted_f1_all_class = weighted_f1_per_class.mean()

	table = wandb.Table(columns=["Class", "F1", "Weighted F1"])
	for i, (f1, weighted_f1) in enumerate(zip(f1_per_class, weighted_f1_per_class)):
		table.add_data(emotion_labels[i], f1.item(), weighted_f1.item())

	wandb.log({
		"Balanced_Accuracy": balanced_acc.item(),
		"Accuracy": accuracy.item(),
		"F1-all-class": weighted_f1_all_class.item(),
		"Classes-with-value": non_zero_count.item(),
		"F1-per-class-bar": wandb.plot.bar(table, "Class", "F1", title="F1 Score per Class"),
		"Weighted-F1-per-class-bar": wandb.plot.bar(table, "Class", "Weighted F1", title="Weighted F1 Score per Class"),
	})

	return {"Accuracy": accuracy, "F1-score": f1_weighted, "F1-all-class": weighted_f1_all_class}


class FocalLoss(nn.Module):
	def __init__(self, alpha=None, gamma=8, ignore_index=-100, reduction="mean"):
		super().__init__()
		# use standard CE loss without reducion as basis
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


y = train_dataset["label"].tolist()
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
class_weights = torch.tensor(class_weights, dtype=torch.float).to("cuda")
loss_fct = FocalLoss(alpha=class_weights, gamma=run.config["focal_gamma"])


class CustomTrainer(Trainer):
	def compute_loss(self, model, inputs, return_outputs=False):
		labels = inputs.get("labels")
		outputs = model(**inputs)
		logits = outputs.get("logits")
		loss = loss_fct(logits, labels)
		return (loss, outputs) if return_outputs else loss


batch_size: int = run.config["batch_size"]

logging_steps: int = len(dataset["train"]) // batch_size
trainer_arguments = TrainingArguments(
	output_dir="./checkpoints",
	overwrite_output_dir=True,
	per_device_train_batch_size=batch_size,
	per_device_eval_batch_size=batch_size,
	gradient_accumulation_steps=1,
	learning_rate=run.config["learning_rate"],
	lr_scheduler_type="cosine",
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
	metric_for_best_model="F1-all-class",
	greater_is_better=True,
	fp16=True,
	bf16=False,
	dataloader_num_workers=12,
	optim=run.config["optim"],
	group_by_length=True,
	report_to=["wandb"],
	hub_model_id=run.config["fine_tuned_model"],
	gradient_checkpointing=True,
	gradient_checkpointing_kwargs={"use_reentrant": True},
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
tuner.push_to_hub()


wandb.finish()
