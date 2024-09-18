import os

import torch
import torch.nn.functional as f
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from libs import EmotionModel, calculate_evaluation_result, representation_evolute
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def sweep(config) -> None:
	run = wandb.init(job_type="Sweep", project="emotion-chat-bot-ncu", group="Emotion Model", config=config)
	dtype = eval(run.config["dtype"])

	# Load Dataset
	dataset = load_dataset(
		"hermeschen1116/emotion_transition_from_dialog",
		num_proc=16,
		trust_remote_code=True,
	)

	model = EmotionModel(dropout=run.config["dropout"], bias=True, dtype=dtype)

	loss_function = torch.nn.CrossEntropyLoss()
	optimizer = eval(f"torch.optim.{run.config['optimizer']}")(model.parameters(), lr=run.config["learning_rate"])

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
		truths: list = []
		predictions: list = []
		model.eval()
		with torch.no_grad():
			for sample in tqdm(validation_dataloader, colour="blue"):
				representation, emotion_composition = (
					sample["bot_representation"],
					sample["user_emotion_composition"],
				)
				labels = f.one_hot(torch.cat(sample["bot_emotion"]), 7).to(dtype)

				output = representation_evolute(model, representation, emotion_composition)

				loss = loss_function(output, labels)
				wandb.log({"val/loss": loss.item()})
				running_loss += loss.item()
				truths += sample["bot_emotion"]
				predictions.append(torch.argmax(output, dim=1))

			wandb.log({"val/loss": running_loss / len(validation_dataloader)})

			evaluation_result: dict = calculate_evaluation_result(torch.cat(predictions), torch.cat(truths))
			wandb.log({"val/f1_score": evaluation_result["f1_score"], "val/accuracy": evaluation_result["accuracy"]})

	model.eval()
	model = torch.compile(model)

	eval_dataset = dataset["test"].map(
		lambda samples: {
			"bot_representation": [
				representation_evolute(model, sample[0], sample[1])
				for sample in zip(
					samples["bot_representation"],
					samples["user_emotion_composition"],
				)
			]
		},
		batched=True,
	)

	eval_dataset = eval_dataset.map(
		lambda samples: {
			"bot_most_possible_emotion": [torch.argmax(torch.tensor(sample), dim=1) for sample in samples]
		},
		input_columns="bot_representation",
		batched=True,
		num_proc=16,
	)

	eval_predictions: Tensor = torch.cat([torch.tensor(turn) for turn in eval_dataset["bot_most_possible_emotion"]])
	eval_truths: Tensor = torch.cat([torch.tensor(turn) for turn in eval_dataset["bot_emotion"]])

	evaluation_result: dict = calculate_evaluation_result(eval_predictions, eval_truths)
	wandb.log(
		{
			"eval/f1-score": evaluation_result["f1_score"],
			"eval/accuracy": evaluation_result["accuracy"],
			"eval/optimize_metric": torch.tensor(list(evaluation_result.values())).dot(torch.tensor([0.5, 0.5])),
		}
	)

	wandb.finish()


sweep_config: dict = {
	"method": "bayes",
	"metric": {"goal": "maximize", "name": "eval/optimize_metric"},
	"parameters": {
		"num_epochs": {"distribution": "int_uniform", "max": 6, "min": 1},
		"dtype": {"values": ["torch.float32", "torch.float16", "torch.bfloat16"]},
		"learning_rate": {"distribution": "uniform", "max": 0.002, "min": 0.0005},
		"dropout": {"distribution": "uniform", "max": 1, "min": 0.25},
		"optimizer": {"values": ["Adagrad", "Adam", "AdamW", "RMSprop", "SGD"]},
	},
}

load_dotenv()
wandb.login(key=os.environ.get("WANDB_API_KEY", ""))
sweep_id = wandb.sweep(sweep_config, project="emotion-chat-bot-ncu")
wandb.agent(sweep_id, sweep, count=100)
