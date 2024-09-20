from argparse import ArgumentParser

import torch
import wandb
from datasets import Array2D, ClassLabel, Sequence, load_dataset
from libs.CommonConfig import CommonScriptArguments, CommonWanDBArguments
from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
	BitsAndBytesConfig,
	HfArgumentParser,
	pipeline,
)

from libs.EmotionTransition import generate_dummy_representation, get_emotion_composition

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
	notes=wandb_args.notes,
	mode=wandb_args.mode,
	resume=wandb_args.resume,
)

dataset = load_dataset("daily_dialog", num_proc=16, trust_remote_code=True).remove_columns(["act"])
emotion_labels: list = dataset["train"].features["emotion"].feature.names
emotion_labels[0] = "neutral"

dataset = dataset.map(
	lambda samples: {"dialog": [[dialog.strip() for dialog in sample] for sample in samples]},
	input_columns="dialog",
	batched=True,
	num_proc=16,
)

dataset = dataset.map(
	lambda samples: {
		"dialog": [sample[:-1] if len(sample) % 2 == 0 else sample for sample in samples["dialog"]],
		"emotion": [sample[:-1] if len(sample) % 2 == 0 else sample for sample in samples["emotion"]],
	},
	batched=True,
	num_proc=16,
)

dataset = dataset.filter(lambda sample: (len(sample["dialog"]) > 2) and (len(sample["emotion"]) > 2), num_proc=16)

dataset = dataset.map(
	lambda samples: {
		"bot_initial_emotion_representation": [
			generate_dummy_representation(sample[0]).unsqueeze(0) for sample in samples["emotion"]
		],
		"bot_emotion": [
			[emotion for i, emotion in enumerate(sample[1:]) if i % 2 == 1] for sample in samples["emotion"]
		],
		"bot_dialog": [[emotion for i, emotion in enumerate(sample[1:]) if i % 2 == 1] for sample in samples["dialog"]],
		"user_dialog": [
			[emotion for i, emotion in enumerate(sample[1:]) if i % 2 == 0] for sample in samples["dialog"]
		],
	},
	remove_columns=["emotion", "dialog"],
	batched=True,
	num_proc=16,
)

dataset = dataset.cast_column("bot_initial_emotion_representation", Array2D((1, 7), "float32"))
dataset = dataset.cast_column("bot_emotion", Sequence(ClassLabel(num_classes=7, names=emotion_labels)))


sentiment_analysis_model = AutoModelForSequenceClassification.from_pretrained(
	run.config["sentiment_analysis_model"],
	quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
	device_map="auto",
	low_cpu_mem_usage=True,
)

sentiment_analysis_tokenizer = AutoTokenizer.from_pretrained(
	run.config["sentiment_analysis_model"], trust_remote_code=True
)

analyser = pipeline(
	"sentiment-analysis",
	model=sentiment_analysis_model,
	tokenizer=sentiment_analysis_tokenizer,
	top_k=7,
	torch_dtype=torch.float32,
	device_map="auto",
	trust_remote_code=True,
)

sentiment_analysis_model = torch.compile(sentiment_analysis_model)


dataset = dataset.map(
	lambda sample: {
		"user_emotion_compositions": [
			[dialog.unsqueeze(0) for dialog in get_emotion_composition(analyser(dialogs)[0])] for dialogs in sample
		]
	},
	input_columns="user_dialog",
)

dataset = dataset.cast_column("user_emotion_compositions", Sequence(Array2D((1, 7), "float32")))


dataset.push_to_hub("emotion_transition_from_dialog", num_shards={"train": 16, "validation": 16, "test": 16})

wandb.finish()
