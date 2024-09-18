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
	representation_evolute,
)
from pyarrow import Tensor
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
from torcheval.metrics.functional.classification import accuracy, f1_score
from tqdm.auto import tqdm
from transformers.hf_argparser import HfArg, HfArgumentParser


@dataclass
class ScriptArguments(CommonScriptArguments):
	dtype: Field[Optional[str]] = HfArg(aliases="--dtype")
	bias: Field[Optional[bool]] = HfArg(aliases="--bias")
	dropout: Field[Optional[float]] = HfArg(aliases="--dropout")
	learning_rate: Field[Optional[float]] = HfArg(aliases="--learning_rate")
	num_epochs: Field[Optional[int]] = HfArg(aliases="--num_epochs")


parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_args()
dtype: torch.dtype = eval(args.dtype)

run = wandb.init(
	job_type="Sweep",
	project="emotion-chat-bot-ncu",
	group="Emotion Model",
)

# Load Dataset
dataset = load_dataset(
	run.config["dataset"],
	num_proc=16,
	trust_remote_code=True,
)

model = EmotionModel(dropout=run.config["dropout"], bias=run.config["bias"], dtype=dtype)

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
		f1_score: Tensor = multiclass_f1_score(
			torch.cat(true_labels),
			torch.cat(predicted_labels),
			num_classes=7,
			average="weighted",
		)
		accuracy = multiclass_accuracy(torch.cat(true_labels), torch.cat(predicted_labels), num_classes=7)
		wandb.log(
			{
				"val/f1_score": accuracy,
				"val/accuracy": f1_score,
			}
		)
		wandb.log({"val/optimize_metric": f1_score * 0.5 + accuracy * 0.5})

model = torch.compile(model)
model.push_to_hub("emotion_model_for_emotion_chat_bot")

wandb.finish()
