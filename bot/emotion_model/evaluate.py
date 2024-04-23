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
from libs.Config import WanDBArguments


@dataclass
class ScriptArgument:
    dtype: Any
    device: str
    huggingface_api_token: str = ""
    wandb_api_token: str = ""


parser = HfArgumentParser((ScriptArgument, WanDBArguments, BitsAndBytesConfig))
args, wandb_args, quantization_config = parser.parse_args()

load_dotenv(encoding="utf-8")
huggingface_hub.login(token=os.environ.get("HF_TOKEN", args.huggingface_api_token), add_to_git_credential=True)
wandb.login(key=os.environ.get("WANDB_API_KEY", args.wandb_api_token), relogin=True)

# Initialize Wandb
run = wandb.init(wandb_args)

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

model = EmotionModel(wandb_args.config["attention_type"], dtype=args.dtype, device=args.device)

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


def get_sentiment_composition(text: str) -> torch.tensor:
    return torch.softmax(torch.tensor([result["score"] for result in analyser(text)]), dim=-1)


evaluation_result: dict = {
    "current_representation": [],
    "input_emotion_composition": [],
    "output_representation": [],
    "predict_emotion": [],
    "true_emotion": []}
for sample in tqdm(eval_dataset, colour="blue"):
    initial_representation: torch.tensor = torch.tensor(sample["initial_representation"],
                                                        device=args.device, dtype=args.dtype)
    turn_result: dict = {
        "current_representation": [],
        "input_emotion_composition": [],
        "output_representation": [],
        "most_possible_emotion": [],
        "true_emotion": []}
    for i, dialog in enumerate(sample["dialog"][:-1]):
        turn_result["current_representation"].append(initial_representation)
        emotion_composition: torch.tensor = get_sentiment_composition(dialog).to(device=args.device, dtype=args.dtype)
        turn_result["input_emotion_composition"].append(emotion_composition)
        new_representation: torch.tensor = model.forward(initial_representation, emotion_composition)
        turn_result["output_representation"].append(new_representation)
        turn_result["most_possible_emotion"].append(torch.argmax(new_representation))
        turn_result["true_emotion"].append(sample["emotion_history"][i + 1])
        initial_representation = new_representation
    for k, v in turn_result:
        evaluation_result[k].append(v)

result = eval_dataset.remove_columns(["initial_representation", "emotion_history"])
for k, v in evaluation_result:
    result = result.add_column(k, v)

emotion_pred: torch.tensor = torch.tensor([emotion for batch in evaluation_result for emotion in batch["most_possible_emotion"]])
emotion_true: torch.tensor = torch.tensor([emotion for batch in evaluation_result for emotion in batch["true_emotion"]])

wandb.log({
    "F1-score": multiclass_f1_score(torch.tensor(emotion_true), torch.tensor(emotion_pred), num_classes=7, average="micro"),
    "Accuracy": multiclass_accuracy(torch.tensor(emotion_true), torch.tensor(emotion_pred), num_classes=7)
})

emotion_labels: list = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
result = result.map(lambda samples: {
    "most_possible_emotion": [[emotion_labels[emotion_id] for emotion_id in sample] for sample in samples]
}, input_columns="most_possible_emotion", batched=True, num_proc=16)
result = result.map(lambda samples: {
    "true_emotion": [[emotion_labels[emotion_id] for emotion_id in sample] for sample in samples]
}, input_columns="ture_emotion", batched=True, num_proc=16)

wandb.log({"evaluation_result": wandb.Table(dataframe=result.to_pandas())})

wandb.finish()
