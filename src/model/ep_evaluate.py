from argparse import ArgumentParser

import torch
from datasets import load_dataset
from libs.CommonConfig import CommonScriptArguments, CommonWanDBArguments
from sklearn.metrics import balanced_accuracy_score, classification_report
from torch import Tensor
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
from transformers import HfArgumentParser, pipeline

import wandb

config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((CommonScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)

run = wandb.init(
	job_type=wandb_args.job_type,
	config=wandb_args.config,
	project=wandb_args.project,
	group=wandb_args.group,
)

dataset = load_dataset(run.config["dataset"], num_proc=16, trust_remote_code=True, split="test")

emotion_labels: list = dataset.features["label"].names
num_emotion_labels: int = len(emotion_labels)

analyser = pipeline(
	model=run.config["model"],
	framework="pt",
	task="sentiment-analysis",
	num_workers=12,
	device_map="auto",
	torch_dtype="auto",
	model_kwargs={
		"low_cpu_mem_usage": True,
	},
	trust_remote_code=True,
)

result = dataset.rename_column("label", "truth_id").add_column("prediction", analyser(dataset["text"]))

id_to_label: dict = {index: label for index, label in enumerate(emotion_labels)}
label_to_id: dict = {label: index for index, label in enumerate(emotion_labels)}
result = result.map(
	lambda samples: {
		"label": [id_to_label[emotion] for emotion in samples["truth_id"]],
		"prediction_id": [label_to_id[emotion["label"]] for emotion in samples["prediction"]],
	},
	batched=True,
	num_proc=16,
)

sentiment_true: Tensor = torch.tensor([sample for sample in result["truth_id"]])
sentiment_pred: Tensor = torch.tensor([sample for sample in result["prediction_id"]])
report = classification_report(
	sentiment_true.tolist(),
	sentiment_pred.tolist(),
	target_names=emotion_labels,
	zero_division=0,
)
print()
print(report)

wandb.log(
	{
		"F1-score": multiclass_f1_score(
			sentiment_pred,
			sentiment_true,
			num_classes=num_emotion_labels,
			average="weighted",
		),
		"Accuracy": multiclass_accuracy(sentiment_pred, sentiment_true, num_classes=num_emotion_labels),
		"Balanced_Accuracy": balanced_accuracy_score(sentiment_true, sentiment_pred),
	}
)
wandb.log({"evaluation_result": wandb.Table(dataframe=result.to_pandas())})

wandb.finish()
