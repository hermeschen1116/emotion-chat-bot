from argparse import ArgumentParser
from dataclasses import Field, dataclass
from typing import Optional

import wandb
from datasets import load_dataset
from libs.CommonConfig import CommonScriptArguments, CommonWanDBArguments
from transformers import HfArgumentParser
from transformers.hf_argparser import HfArg


@dataclass
class ScriptArguments(CommonScriptArguments):
	dataset_name: Field[Optional[str]] = HfArg(aliases="--dataset-name", default="daily_dialog_for_RG")
	dataset_description: Field[Optional[str]] = HfArg(aliases="--dataset-description", default="")


config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)

run = wandb.init(
	job_type=wandb_args.job_type,
	config=wandb_args.config,
	project=wandb_args.project,
	group=wandb_args.group,
	notes=wandb_args.notes,
	mode=wandb_args.mode,
	resume=wandb_args.resume,
)

dataset = load_dataset("daily_dialog", num_proc=16, save_infos=True, trust_remote_code=True).remove_columns("act")

dataset = dataset.rename_column("emotion", "emotion_id")
emotion_labels: list = dataset["train"].features["emotion_id"].feature.names
emotion_labels[0] = "neutral"
dataset = dataset.map(
	lambda samples: {"emotion": [[emotion_labels[emotion_id] for emotion_id in sample] for sample in samples]},
	input_columns="emotion_id",
	remove_columns="emotion_id",
	batched=True,
	num_proc=16,
)

dataset = dataset.map(
	lambda samples: {
		"emotion": [sample[:-1] if len(sample) % 2 == 1 else sample for sample in samples["emotion"]],
		"dialog": [sample[:-1] if len(sample) % 2 == 1 else sample for sample in samples["dialog"]],
	},
	batched=True,
	num_proc=16,
)

dataset = dataset.filter(lambda sample: (len(sample["emotion"]) != 0) and (len(sample["dialog"]) != 0), num_proc=16)

dataset = dataset.map(
	lambda samples: {
		"prompt": [
			[
				{
					"role": "user" if i % 2 == 0 else "assistant",
					"content": {"emotion": emotion, "dialog": dialog.strip()},
				}
				for i, (emotion, dialog) in enumerate(zip(sample[0], sample[1]))
			]
			for sample in zip(samples["emotion"], samples["dialog"])
		]
	},
	remove_columns=["emotion", "dialog"],
	batched=True,
	num_proc=16,
)

dataset.push_to_hub("daily_dialog_for_RG", num_shards={"train": 16, "validation": 16, "test": 16})

wandb.finish()
