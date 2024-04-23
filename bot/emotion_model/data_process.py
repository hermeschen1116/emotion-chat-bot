import os
import sys
from typing import Optional

import torch
from transformers.hf_argparser import HfArg

from bot.emotion_model.libs.DataProcess import get_sentiment_composition, generate_dummy_representation

sys.path.append("../../../chat-bot")

from libs.CommonConfig import CommonWanDBArguments, CommonScriptArguments

import shutil
from dataclasses import dataclass

import huggingface_hub
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (HfArgumentParser,
                          BitsAndBytesConfig,
                          AutoModelForSequenceClassification,
                          AutoTokenizer,
                          pipeline)


@dataclass
class ScriptArguments(CommonScriptArguments):
    dataset_path: Optional[str] = HfArg(aliases="--dataset-path", default="./dataset")


parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments, BitsAndBytesConfig))
args, wandb_args, quantization_config, remain_args = parser.parse_args()

load_dotenv(encoding="utf-8")
huggingface_hub.login(token=os.environ.get("HF_TOKEN", args.huggingface_api_token), add_to_git_credential=True)
wandb.login(key=os.environ.get("WANDB_API_KEY", args.wandb_api_token), relogin=True)

run = wandb.init(
    job_type=wandb_args.job_type,
    config=wandb_args.config,
    project=wandb_args.project,
    group=wandb_args.group,
    notes=wandb_args.notes,
    mode=wandb_args.mode,
    resume=wandb_args.resume
)

dataset = load_dataset("daily_dialog",
                       num_proc=16,
                       trust_remote_code=True).remove_columns(["act"])

dataset = dataset.map(lambda samples: {
    "dialog": [[dialog.strip() for dialog in sample] for sample in samples]
}, input_columns="dialog", batched=True, num_proc=16)

dataset = dataset.map(lambda samples: {
    "dialog": [sample[:-1] if len(sample) % 2 == 0 else sample for sample in samples["dialog"]],
    "emotion": [sample[:-1] if len(sample) % 2 == 0 else sample for sample in samples["emotion"]]
}, batched=True, num_proc=16)

eval_dataset = dataset.map(lambda samples: {
    "user_representation": [[generate_dummy_representation(sample[0])] for sample in samples["emotion"]],
    "user_emotion": [[emotion for i, emotion in enumerate(sample[1:]) if i % 2 == 1] for sample in samples["emotion"]],
    "user_dialog": [[emotion for i, emotion in enumerate(sample[1:]) if i % 2 == 1] for sample in samples["dialog"]],
    "bot_dialog": [[emotion for i, emotion in enumerate(sample[1:]) if i % 2 == 0] for sample in samples["dialog"]]
}, remove_columns=["emotion", "dialog"], batched=True, num_proc=16)

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16
# )
quantization_config = None if not torch.cuda.is_available() else quantization_config
sentiment_analysis_model = AutoModelForSequenceClassification.from_pretrained(
    wandb_args.config["sentiment_analysis_model"],
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2",
    device_map="auto",
    low_cpu_mem_usage=True)

sentiment_analysis_tokenizer = AutoTokenizer.from_pretrained(
    wandb_args.config["sentiment_analysis_tokenizer"],
    trust_remote_code=True)

analyser = pipeline(
    "sentiment-analysis",
    model=sentiment_analysis_model,
    tokenizer=sentiment_analysis_tokenizer,
    top_k=0,
    device_map="auto",
    trust_remote_code=True)

sentiment_analysis_model = torch.compile(sentiment_analysis_model)

eval_dataset = eval_dataset.map(lambda samples: {
    "bot_dialog_emotion_composition": [[get_sentiment_composition(analyser(dialog))
                                        for dialog in sample] for sample in samples]
}, input_columns="bot_dialog", batched=True, num_proc=16)

dataset_artifact = wandb.Artifact(
    "daily_dialog_for_EM",
    type="dataset",
    description="modified version of daily dialog dataset from huggingface for emotion model module",
    metadata=dict(dataset),
    incremental=True
)

dataset.save_to_disk(args.dataset_path, num_proc=16)
dataset_artifact.add_dir(args.dataset_path)

run.log_artifact(dataset_artifact)

wandb.finish()

if os.path.exists(args.dataset_path):
    shutil.rmtree(args.dataset_path, ignore_errors=True)
