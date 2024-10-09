from argparse import ArgumentParser

import torch
import wandb
from datasets import Dataset, load_dataset
from torch import Tensor
from transformers.hf_argparser import HfArgumentParser

from emotion_chat_bot.model.emotion_model.EmotionTransition import EmotionModel, representation_evolute
from emotion_chat_bot.utils.CommonConfig import CommonScriptArguments, CommonWanDBArguments
from emotion_chat_bot.utils.CommonUtils import calculate_evaluation_result, get_torch_device

config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((CommonScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)
device: str = get_torch_device()

# Initialize Wandb
run = wandb.init(
	job_type=wandb_args.job_type,
	config=wandb_args.config,
	project=wandb_args.project,
	group=wandb_args.group,
	notes=wandb_args.notes,
	mode=wandb_args.mode,
	resume=wandb_args.resume,
)

eval_dataset: Dataset = load_dataset(
	run.config["dataset"], split="test", num_proc=16, trust_remote_code=True
).remove_columns(["bot_dialog", "user_dialog"])

emotion_labels: list = eval_dataset.features["bot_emotion"].feature.names

model = EmotionModel.from_pretrained(run.config["model"]).to(device)

eval_dataset = eval_dataset.map(
	lambda samples: {
		"bot_emotion_representations": [
			representation_evolute(
				model,
				[torch.tensor(sample[0][0]).to(device)],
				[torch.tensor(emotion).to(device) for emotion in sample[1]],
			)[1:]
			for sample in zip(samples["bot_initial_emotion_representation"], samples["user_emotion_compositions"])
		]
	},
	remove_columns=["bot_initial_emotion_representation"],
	batched=True,
)

eval_dataset = eval_dataset.map(
	lambda samples: {"bot_possible_emotion": [torch.tensor(sample).argmax(1) for sample in samples]},
	input_columns=["bot_emotion_representations"],
	batched=True,
	num_proc=16,
)

eval_predictions: Tensor = torch.cat([torch.tensor(turn) for turn in eval_dataset["bot_possible_emotion"]])
eval_truths: Tensor = torch.cat([torch.tensor(turn) for turn in eval_dataset["bot_emotion"]])

metrics_result: dict = calculate_evaluation_result(eval_predictions, eval_truths)
wandb.log({"F1-score": metrics_result["f1_score"], "Accuracy": metrics_result["accuracy"]})

wandb.finish()
