import json
from argparse import ArgumentParser

import torch
import torch.nn.functional as f
import wandb
from datasets import load_dataset
from libs import EmotionModel, calculate_evaluation_result, get_torch_device, login_to_service, representation_evolute
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def sweep_function(config: dict = None) -> None:
	run = wandb.init(job_type="Sweep", project="emotion-chat-bot-ncu", group="Emotion Model", config=config)
	device: str = get_torch_device()
	dtype = eval(run.config["dtype"])

	# Load Dataset
	dataset = load_dataset(
		"hermeschen1116/emotion_transition_from_dialog",
		num_proc=16,
		keep_in_memory=True,
		trust_remote_code=True,
	)

	dataset.set_format("torch")

	model = EmotionModel(
		attention=run.config["attention"], dropout=run.config["dropout"], bias=run.config["bias"], dtype=dtype
	).to(device)

	loss_function = torch.nn.CrossEntropyLoss()
	optimizer = eval(f"torch.optim.{run.config['optimizer']}")(model.parameters(), lr=run.config["learning_rate"])

	train_dataloader = DataLoader(
		dataset["train"], batch_size=1, shuffle=True, num_workers=16, pin_memory=True, pin_memory_device=device
	)
	validation_dataloader = DataLoader(
		dataset["validation"], batch_size=1, shuffle=True, num_workers=16, pin_memory=True, pin_memory_device=device
	)
	for i in range(run.config["num_epochs"]):
		running_loss: float = 0
		model.train()
		for sample in tqdm(train_dataloader, colour="green"):
			representations: list = [sample["bot_initial_emotion_representation"][0].to(device)]
			compositions: list = [
				emotion.transpose(1, 0) for emotion in sample["user_emotion_compositions"][0].to(device)
			]
			labels: Tensor = f.one_hot(sample["bot_emotion"][0], 7).to(device)

			optimizer.zero_grad()

			output: Tensor = (
				torch.tensor(representation_evolute(model, representations, compositions)).argmax(1).to(device)
			)

			loss = loss_function(output, labels)
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
				labels: Tensor = f.one_hot(sample["bot_emotion"][0], 7).to(device)

				output: Tensor = (
					torch.tensor(representation_evolute(model, representations, compositions)).argmax(1).to(device)
				)

				loss = loss_function(output, labels)
				wandb.log({"val/loss": loss.item()})
				running_loss += loss.item()
				truths += sample["bot_emotion"].tolist()
				predictions += torch.argmax(output, dim=1).tolist()

			wandb.log({"val/loss": running_loss / len(validation_dataloader)})

			evaluation_result: dict = calculate_evaluation_result(torch.tensor(predictions), torch.tensor(truths))
			wandb.log({"val/f1_score": evaluation_result["f1_score"], "val/accuracy": evaluation_result["accuracy"]})

	model.eval()
	model = torch.compile(model)

	eval_dataset = dataset["test"].map(
		lambda samples: {
			"bot_emotion_representations": [
				representation_evolute(
					model, [sample[0][0].to(device)], [emotion.transpose(1, 0) for emotion in sample[1][0].to(device)]
				)
				for sample in zip(
					samples["bot_initial_emotion_representation"],
					samples["user_emotion_compositions"],
				)
			]
		},
		batched=True,
	)

	eval_dataset = eval_dataset.map(
		lambda samples: {"bot_possible_emotion": [torch.tensor(sample).argmax(1) for sample in samples]},
		input_columns="bot_representation",
		batched=True,
		num_proc=16,
	)

	eval_predictions: Tensor = torch.cat([torch.tensor(turn) for turn in eval_dataset["bot_possible_emotion"]])
	eval_truths: Tensor = torch.cat([torch.tensor(turn) for turn in eval_dataset["bot_emotion"]])

	evaluation_result: dict = calculate_evaluation_result(eval_predictions, eval_truths)
	wandb.log(
		{
			"eval/f1-score": evaluation_result["f1_score"],
			"eval/accuracy": evaluation_result["accuracy"],
			"eval/optimize_metric": torch.tensor(list(evaluation_result.values())).dot(torch.tensor([0.5, 0.5])),
		}
	)


login_to_service()
config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

with open(config.json_file, "r", encoding="utf-8") as config_file:
	sweep_config: dict = json.load(config_file)

sweep_id = wandb.sweep(sweep=sweep_config, project="emotion-chat-bot-ncu")
wandb.agent(sweep_id, sweep_function, project="emotion-chat-bot-ncu", count=100)
wandb.finish()
