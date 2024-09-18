from argparse import ArgumentParser
from dataclasses import Field, dataclass
from typing import Optional

import torch
import torch.nn.functional as f
import wandb
from datasets import load_dataset
from libs import (
	CommonScriptArguments,
	CommonWanDBArguments,
	EmotionModel,
	get_torch_device,
	representation_evolute,
)
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
from tqdm.auto import tqdm
from transformers.hf_argparser import HfArg, HfArgumentParser


@dataclass
class ScriptArguments(CommonScriptArguments):
	dtype: Field[Optional[str]] = HfArg(aliases="--dtype", default="torch.float32")
	device: Field[Optional[str]] = HfArg(aliases="--device", default_factory=get_torch_device)


config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)
dtype: torch.dtype = eval(args.dtype)

run = wandb.init(
	job_type=wandb_args.job_type,
	config=wandb_args.config,
	project=wandb_args.project,
	group=wandb_args.group,
	notes=wandb_args.notes,
	mode=wandb_args.mode,
	resume=wandb_args.resume,
)

# Load Dataset
dataset = load_dataset(
	run.config["dataset"],
	num_proc=16,
	trust_remote_code=True,
)

model = EmotionModel(
	dropout=run.config["dropout"],
	bias=run.config["bias"],
	dtype=dtype
)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=run.config["learning_rate"])

train_dataloader = DataLoader(dataset["train"])
validation_dataloader = DataLoader(dataset["validation"])
for i in range(run.config["num_epochs"]):
	running_loss: float = 0
	model.train()
	for sample in tqdm(train_dataloader, colour="green"):
		representation, emotion_composition = (
			sample["bot_representation"],
			sample["user_emotion_composition"],
		)
		labels = f.one_hot(torch.cat(sample["bot_emotion"]), 7).to(dtype)

		optimizer.zero_grad()

		output = representation_evolute(model, representation, emotion_composition)

		loss = loss_function(output, labels)
		wandb.log({"train/loss": loss.item()})
		running_loss += loss.item()

		loss.backward()
		optimizer.step()

	if i + 1 == run.config["num_epochs"]:
		wandb.log({"train/train_loss": running_loss / len(train_dataloader)})

	running_loss = 0
	true_labels: list = []
	predicted_labels: list = []
	model.eval()
	with torch.no_grad():
		for sample in tqdm(validation_dataloader, colour="blue"):
			representation, emotion_composition = (
				sample["bot_representation"],
				sample["user_emotion_composition"],
			)
			labels = f.one_hot(torch.cat(sample["bot_emotion"]), 7).to(dtype=torch.float)

			output = representation_evolute(model, representation, emotion_composition)

			loss = loss_function(output, labels)
			wandb.log({"val/loss": loss.item()})
			running_loss += loss.item()
			true_labels += sample["bot_emotion"]
			predicted_labels.append(torch.argmax(output, dim=1))

		wandb.log({"val/val_loss": running_loss / len(validation_dataloader)})
		wandb.log(
			{
				"val/f1_score": multiclass_f1_score(
					torch.cat(true_labels),
					torch.cat(predicted_labels),
					num_classes=7,
					average="weighted",
				),
				"val/accuracy": multiclass_accuracy(torch.cat(true_labels), torch.cat(predicted_labels), num_classes=7),
			}
		)

model = torch.compile(model)
model.push_to_hub("emotion_model_for_emotion_chat_bot")

wandb.finish()
