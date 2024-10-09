from argparse import ArgumentParser
from dataclasses import Field, dataclass

import torch
import wandb
from datasets import load_dataset
from peft.peft_model import PeftModel
from torch import Tensor
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
from transformers import (
	BitsAndBytesConfig,
	GenerationConfig,
	HfArgumentParser,
	TextStreamer,
	pipeline,
)
from transformers.hf_argparser import HfArg
from unsloth import FastLanguageModel

from emotion_chat_bot.pipeline.ResponseGenerationPipeline import ResponseGeneratorPipeline
from emotion_chat_bot.utils.CommonConfig import CommonScriptArguments, CommonWanDBArguments


@dataclass
class ScriptArguments(CommonScriptArguments):
	chat_template_file: Field[str] = HfArg(aliases="--chat-template-file", default="")


config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)

chat_template: dict = eval(open(args.chat_template_file, "r", encoding="utf-8", closefd=True).read())

run = wandb.init(
	job_type=wandb_args.job_type,
	config=wandb_args.config,
	project=wandb_args.project,
	group=wandb_args.group,
	notes=wandb_args.notes,
	mode=wandb_args.mode,
	resume=wandb_args.resume,
)
run.config["chat_template"] = chat_template["template"]
run.config["instruction_template"] = chat_template["instruction"]
run.config["response_template"] = chat_template["response"]
run.config["special_tokens"] = chat_template["special_tokens"]


# Load and Process Dataset
dataset = load_dataset(
	"hermeschen1116/daily_dialog_for_RG",
	split="test",
	num_proc=16,
	trust_remote_code=True,
)
# dataset = dataset.train_test_split(test_size=0.001)["test"]

dataset = dataset.map(
	lambda samples: {
		"dialog_bot": [sample[-1]["content"]["dialog"] for sample in samples],
		"emotion_bot": [sample[-1]["content"]["emotion"] for sample in samples],
	},
	input_columns="prompt",
	batched=True,
	num_proc=16,
)

dataset = dataset.map(
	lambda samples: {
		"history": [
			"\n".join(
				[f"{turn['role']}({turn['content']['emotion']}): {turn['content']['dialog']}" for turn in sample[:-1]]
			)
			for sample in samples
		]
	},
	input_columns="prompt",
	batched=True,
	num_proc=16,
)

system_prompt: list = [
	{
		"role": "system",
		"content": {"emotion": "", "dialog": run.config["system_prompt"]},
	}
]

dataset = dataset.map(
	lambda samples: {"prompt": [system_prompt + sample for sample in samples]},
	input_columns="prompt",
	batched=True,
	num_proc=16,
)

# Load Tokenizer
base_model, tokenizer = FastLanguageModel.from_pretrained(
	run.config["base_model"],
	attn_implementation="flash_attention_2",
	pretraining_tp=1,
	load_in_4bit=True,
	device_map="auto",
	low_cpu_mem_usage=True,
	trust_remote_code=True,
)
tokenizer.padding_side = "left"
tokenizer.clean_up_tokenization_spaces = True
tokenizer.chat_template = run.config["chat_template"]
tokenizer.add_special_tokens(run.config["special_tokens"])
base_model.resize_token_embeddings(len(tokenizer))

run.config["example_prompt"] = tokenizer.apply_chat_template(dataset[0]["prompt"], tokenize=False)

model = PeftModel.from_pretrained(base_model, run.use_model(run.config["fine_tuned_model"]))
model = torch.compile(model)
FastLanguageModel.for_inference(model)

# Generate Response
bot = ResponseGeneratorPipeline(
	model,
	tokenizer,
	framework="pt",
	task="conversation-generation",
	num_workers=16,
	torch_dtype="auto",
	add_special_tokens=True,
	truncation=False,
	padding=True,
)

streamer = TextStreamer(tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=True)

generation_config = GenerationConfig(
	max_new_tokens=20,
	min_new_tokens=5,
	repetition_penalty=1.5,
	pad_token_id=tokenizer.pad_token_id,
	eos_token_id=tokenizer.eos_token_id,
)

result = dataset.map(
	lambda sample: {
		"test_response": bot(sample, streamer=streamer, generation_config=generation_config)[0]["generated_text"][-1][
			"content"
		]["dialog"]
	},
	input_columns="prompt",
)
result = result.remove_columns("prompt")

# Sentiment Analysis
analyser = pipeline(
	model="Shotaro30678/sentiment_analysis_for_emotion_chat_bot",
	framework="pt",
	task="sentiment-analysis",
	num_workers=16,
	device_map="auto",
	torch_dtype="auto",
	model_kwargs={
		"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
		"low_cpu_mem_usage": True,
	},
	trust_remote_code=True,
)

result = result.add_column("test_response_sentiment", analyser(result["test_response"]))

# Metrics
emotion_labels: list = [
	"neutral",
	"anger",
	"disgust",
	"fear",
	"happiness",
	"sadness",
	"surprise ",
]

emotion_id: dict = {label: index for index, label in enumerate(emotion_labels)}

sentiment_true: Tensor = torch.tensor([emotion_id[sample] for sample in result["emotion_bot"]])
sentiment_pred: Tensor = torch.tensor([emotion_id[sample["label"]] for sample in result["test_response_sentiment"]])

num_emotion_labels: int = len(emotion_labels)
wandb.log(
	{
		"F1-score": multiclass_f1_score(
			sentiment_pred,
			sentiment_true,
			num_classes=num_emotion_labels,
			average="weighted",
		),
		"Accuracy": multiclass_accuracy(sentiment_pred, sentiment_true, num_classes=num_emotion_labels),
	}
)
wandb.log({"evaluation_result": wandb.Table(dataframe=result.to_pandas())})

wandb.finish()
