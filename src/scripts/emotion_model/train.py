from argparse import ArgumentParser

import torch
import torch.nn.functional as f
import wandb
from datasets import load_dataset
from datasets.load import DatasetDict
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.hf_argparser import HfArgumentParser

from emotion_chat_bot.model.emotion_model.EmotionTransition import EmotionModel, representation_evolute
from emotion_chat_bot.utils.Config import CommonScriptArguments, CommonWanDBArguments
from emotion_chat_bot.utils.model.Evaluate import calculate_evaluation_result
from emotion_chat_bot.utils.model.Train import get_torch_device

config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((CommonScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)
device: str = get_torch_device()

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
dataset: DatasetDict = load_dataset(run.config["dataset"], num_proc=16, keep_in_memory=True, trust_remote_code=True)

model = EmotionModel().to(device)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=run.config["learning_rate"])

train_dataloader = DataLoader(
	dataset["train"].with_format("torch"),
	batch_size=1,
	shuffle=True,
	num_workers=12,
	pin_memory=True,
	pin_memory_device=device,
)
validation_dataloader = DataLoader(
	dataset["validation"].with_format("torch"),
	batch_size=1,
	shuffle=True,
	num_workers=12,
	pin_memory=True,
	pin_memory_device=device,
)
for i in range(run.config["num_epochs"]):
	running_loss: float = 0
	model.train()
	for sample in tqdm(train_dataloader, colour="green"):
		representations: list = [sample["bot_initial_emotion_representation"][0].to(device)]
		compositions: list = [emotion.transpose(1, 0) for emotion in sample["user_emotion_compositions"][0].to(device)]
		labels: Tensor = f.one_hot(sample["bot_emotion"][0], 7).float().to(device)

		optimizer.zero_grad()

		outputs: Tensor = (
			torch.stack(representation_evolute(model, representations, compositions)[1:], dim=1)[0].float().to(device)
		)

		loss = loss_function(outputs, labels)
		wandb.log({"train/loss": loss.item()})
		running_loss += loss.item()

		loss.backward()
		optimizer.step()

	if i + 1 == run.config["num_epochs"]:
		wandb.log({"train/train_loss": running_loss / len(train_dataloader)})

	running_loss = 0
	truths: list = []
	predictions: list = []
	model.eval()
	with torch.no_grad():
		for sample in tqdm(validation_dataloader, colour="blue"):
			representations: list = [sample["bot_initial_emotion_representation"][0].to(device)]
			compositions: list = [
				emotion.transpose(1, 0) for emotion in sample["user_emotion_compositions"][0].to(device)
			]
			labels: Tensor = f.one_hot(sample["bot_emotion"][0], 7).float().to(device)

			outputs: Tensor = (
				torch.stack(representation_evolute(model, representations, compositions)[1:], dim=1)[0]
				.float()
				.to(device)
			)

			loss = loss_function(outputs, labels)
			wandb.log({"val/loss": loss.item()})
			running_loss += loss.item()
			truths += sample["bot_emotion"].tolist()[0]
			predictions += torch.argmax(outputs, dim=1).tolist()

		wandb.log({"val/loss": running_loss / len(validation_dataloader)})
		evaluation_result: dict = calculate_evaluation_result(torch.tensor(predictions), torch.tensor(truths))
		wandb.log({"val/f1_score": evaluation_result["f1_score"], "val/accuracy": evaluation_result["accuracy"]})

model = torch.compile(model)
model.push_to_hub(run.config["trained_model_name"])

wandb.finish()
