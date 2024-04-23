import os
from dataclasses import dataclass
from typing import Any

import huggingface_hub
import torch
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline, HfArgumentParser
)

from EmotionModel import EmotionModel


@dataclass
class ScriptArgument:
    dtype: Any
    device: str
    attention_type: str
    huggingface_api_token: str = ""
    wandb_api_token: str = ""


parser = HfArgumentParser((ScriptArgument, WanDBArguments, BitsAndBytesConfig))
args, wandb_args, quantization_config = parser.parse_args()

load_dotenv(encoding="utf-8")

huggingface_hub.login(token=os.environ.get("HF_TOKEN", args.huggingface_api_token), add_to_git_credential=True)
wandb.login(key=os.environ.get("WANDB_API_KEY", args.wandb_api_token), relogin=True)

dataset = load_dataset("daily_dialog", split="test", num_proc=16, trust_remote_code=True).remove_columns(["act"])
dataset = dataset.map(lambda samples: {
    "dialog": [[dialog.strip() for dialog in sample] for sample in samples]
}, input_columns="dialog", batched=True, num_proc=16)
dataset = dataset.map(lambda samples: {
    "dialog": [sample[:-1] if len(sample) % 2 == 0 else sample for sample in samples["dialog"]],
    "emotion": [sample[:-1] if len(sample) % 2 == 0 else sample for sample in samples["emotion"]]
}, batched=True, num_proc=16)


def generate_dummy_representation(target_emotion: int) -> torch.tensor:
    while True:
        dummy = torch.clamp(torch.rand(7, dtype=args.dtype), -1, 1)
        if torch.argmax(dummy) == target_emotion:
            return dummy


eval_dataset = dataset.map(lambda samples: {
    "initial_representation": [generate_dummy_representation(sample[0]) for sample in samples],
    "emotion_history": [sample[1:] for sample in samples]
}, input_columns="emotion", remove_columns="emotion", batched=True, num_proc=16)

model = EmotionModel(args.attention_type, dtype=args.dtype, device=args.device)

# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16
# )
quantization_config = None if not torch.cuda.is_available() else quantization_config
sentiment_analysis_model = AutoModelForSequenceClassification.from_pretrained(
    "../sentiment_analysis/emotion_text_classifier_on_dd_v1",
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2",
    device_map="auto",
    low_cpu_mem_usage=True)

sentiment_analysis_tokenizer = AutoTokenizer.from_pretrained(
    "michellejieli/emotion_text_classifier",
    trust_remote_code=True)

analyser = pipeline(
    "sentiment-analysis",
    model=sentiment_analysis_model,
    tokenizer=sentiment_analysis_tokenizer,
    top_k=0,
    device_map="auto",
    trust_remote_code=True)

sentiment_analysis_model = torch.compile(sentiment_analysis_model)


def get_sentiment_composition(text: str) -> torch.tensor:
    return torch.softmax(torch.tensor([result["score"] for result in analyser(text)]), dim=-1)


emotion_pred: list = []
emotion_true: list = []

for sample in tqdm(eval_dataset, colour="blue"):
    initial_representation: torch.tensor = torch.tensor(sample["initial_representation"],
                                                        device=args.device, dtype=args.dtype)
    for i, dialog in enumerate(sample["dialog"][:-1]):
        emotion_composition: torch.tensor = get_sentiment_composition(dialog).to(device=args.device, dtype=args.dtype)
        new_representation: torch.tensor = model.forward(initial_representation, emotion_composition)
        emotion_pred.append(torch.argmax(new_representation))
        emotion_true.append(sample["emotion_history"][i + 1])

emotion_pred: torch.tensor = torch.tensor(emotion_pred)
emotion_true: torch.tensor = torch.tensor(emotion_true)

f1 = multiclass_f1_score(emotion_true, emotion_pred, num_classes=7, average="micro")

accuracy = multiclass_accuracy(emotion_true, emotion_pred, num_classes=7)
