from argparse import ArgumentParser

import torch
import wandb
from datasets import load_dataset
from libs import (
    CommonScriptArguments,
    CommonWanDBArguments,
    flatten_data_and_abandon_data_with_neutral,
)
from torch import Tensor
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
from transformers import HfArgumentParser, pipeline

config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((CommonScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)

run = wandb.init(
    name=wandb_args.name,
    job_type=wandb_args.job_type,
    config=wandb_args.config,
    project=wandb_args.project,
    group=wandb_args.group,
    notes=wandb_args.notes,
)

dataset = load_dataset(
    run.config["dataset"],
    split="test",
    num_proc=16,
    trust_remote_code=True,
).remove_columns(["act"])
emotion_labels: list = dataset.features["emotion"].feature.names
emotion_labels[0] = "neutral"
num_emotion_labels: int = len(emotion_labels)

dataset = dataset.map(
    lambda samples: {
        "response_emotion": [sample[1:] + [sample[0]] for sample in samples]
    },
    input_columns=["emotion"],
    remove_columns=["emotion"],
    batched=True,
    num_proc=16,
)

dataset = dataset.map(
    lambda samples: {
        "dialog": [sample[:-1] for sample in samples["dialog"]],
        "response_emotion": [sample[:-1] for sample in samples["response_emotion"]],
    },
    batched=True,
    num_proc=16,
)

dataset = dataset.map(
    lambda samples: {
        "dialog": [sample[:-1] for sample in samples["dialog"]],
        "response_emotion": [sample[:-1] for sample in samples["response_emotion"]],
    },
    batched=True,
    num_proc=16,
)

dataset = dataset.map(
    lambda samples: {
        "rows": [
            [
                {
                    "text": dialog,
                    "label": emotion,
                }
                for i, (emotion, dialog) in enumerate(zip(sample[0], sample[1]))
            ]
            for sample in zip(samples["response_emotion"], samples["dialog"])
        ]
    },
    remove_columns=["response_emotion", "dialog"],
    batched=True,
    num_proc=16,
)
dataset = dataset.map(
    lambda samples: {
        "rows": [sample for sample in samples if len(sample) != 0],
    },
    input_columns=["rows"],
    batched=True,
    num_proc=16,
)

dataset = flatten_data_and_abandon_data_with_neutral(dataset, 1)

analyser = pipeline(
    model="hermeschen1116/emotion_predictor_for_emotion_chat_bot",
    framework="pt",
    task="sentiment-analysis",
    num_workers=16,
    device_map="auto",
    torch_dtype="auto",
    model_kwargs={
        "low_cpu_mem_usage": True,
    },
    trust_remote_code=True,
)

result = dataset.rename_column("label", "truth_id").add_column(
    "prediction_id", analyser(dataset["text"])
)

emotion_label: dict = {index: label for index, label in enumerate(emotion_labels)}
result = result.map(
    lambda samples: {
        "label": [
            emotion_label[emotion_id]
            for sample in samples["truth_id"]
            for emotion_id in sample
        ],
        "prediction": [
            emotion_label[emotion_id]
            for sample in samples["prediction_id"]
            for emotion_id in sample
        ],
    },
    batched=True,
    num_proc=16,
)

sentiment_true: Tensor = torch.tensor([sample for sample in result["truth_id"]])
sentiment_pred: Tensor = torch.tensor([sample for sample in result["prediction_id"]])

wandb.log(
    {
        "F1-score": multiclass_f1_score(
            sentiment_true,
            sentiment_pred,
            num_classes=num_emotion_labels,
            average="weighted",
        ),
        "Accuracy": multiclass_accuracy(
            sentiment_true, sentiment_pred, num_classes=num_emotion_labels
        ),
    }
)
wandb.log({"evaluation_result": wandb.Table(dataframe=result.to_pandas())})

wandb.finish()
