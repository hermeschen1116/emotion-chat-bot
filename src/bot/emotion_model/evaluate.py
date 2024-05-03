from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Optional

import torch
import wandb
from datasets import load_from_disk
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
from transformers.hf_argparser import HfArgumentParser, HfArg

from libs.CommonConfig import CommonWanDBArguments, CommonScriptArguments, get_torch_device
from libs.EmotionModel import EmotionModel


@dataclass
class ScriptArguments(CommonScriptArguments):
    dtype: Optional[Any] = HfArg(aliases="--dtype", default=torch.float32)
    device: Optional[str] = HfArg(aliases="--device", default_factory=get_torch_device)


config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)

# Initialize Wandb
run = wandb.init(
    job_type=wandb_args.job_type,
    config=wandb_args.config,
    project=wandb_args.project,
    group=wandb_args.group,
    notes=wandb_args.notes,
    mode=wandb_args.mode,
    resume=wandb_args.resume
)

dataset_path = run.use_artifact("daily_dialog_for_EM:latest").download()
eval_dataset = load_from_disk(dataset_path)["test"]

model = EmotionModel(wandb_args.config["attention_type"], dtype=args.dtype)
model = torch.compile(model)

eval_dataset = eval_dataset.map(lambda samples: {
    "bot_representation": [model.representation_evolute(sample[0], sample[1])
                           for sample in zip(samples["bot_representation"],
                                             samples["user_dialog_emotion_composition"])]
}, batched=True)

eval_dataset = eval_dataset.map(lambda samples: {
    "bot_most_possible_emotion": [torch.argmax(torch.tensor(sample), dim=1) for sample in samples]
}, input_columns="bot_representation", batched=True, num_proc=16)

predicted_labels: torch.tensor = torch.cat([torch.tensor(turn) for turn in eval_dataset["bot_most_possible_emotion"]])
true_labels: torch.tensor = torch.cat([torch.tensor(turn) for turn in eval_dataset["bot_emotion"]])

wandb.log({
    "F1-score": multiclass_f1_score(true_labels, predicted_labels, num_classes=7, average="weighted"),
    "Accuracy": multiclass_accuracy(true_labels, predicted_labels, num_classes=7)
})

emotion_labels: list = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
eval_dataset = eval_dataset.map(lambda samples: {
    "bot_most_possible_emotion": [[emotion_labels[emotion_id] for emotion_id in sample]
                                  for sample in samples["bot_most_possible_emotion"]],
    "bot_emotion": [[emotion_labels[emotion_id] for emotion_id in sample] for sample in samples["bot_emotion"]]
}, batched=True, num_proc=16)

result = eval_dataset.map(lambda samples: {
    "bot_most_possible_emotion": [", ".join(sample) for sample in samples["bot_most_possible_emotion"]],
    "bot_emotion": [", ".join(sample) for sample in samples["bot_emotion"]]
}, batched=True, num_proc=16)

result = result.remove_columns(["bot_representation", "bot_dialog", "user_dialog", "user_dialog_emotion_composition"])

wandb.log({"evaluation_result": wandb.Table(dataframe=result.to_pandas())})

wandb.finish()
