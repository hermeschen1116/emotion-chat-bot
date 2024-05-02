from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as f
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers.hf_argparser import HfArgumentParser, HfArg

import wandb
from libs.CommonConfig import (
    CommonWanDBArguments,
    CommonScriptArguments,
    get_torch_device,
)
from libs.EmotionModel import EmotionModel


@dataclass
class ScriptArguments(CommonScriptArguments):
    dataset: Optional[str] = HfArg(aliases="--dataset", default="daily_dialog_for_EM:latest")
    dtype: Optional[Any] = HfArg(aliases="--dtype", default=torch.float32)
    device: Optional[str] = HfArg(aliases="--device", default_factory=get_torch_device)


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
    resume=wandb_args.resume
)

# Load Dataset
dataset_path = run.use_artifact(args.dataset).download()
dataset = load_from_disk(dataset_path)

model = EmotionModel(wandb.config["attention_type"])
model = torch.compile(model)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

train_dataloader = DataLoader(dataset["train"])
validation_dataloader = DataLoader(dataset["validation"])

for i in range(wandb.config["num_epochs"]):
    model.train()
    for sample in tqdm(train_dataloader, colour="green"):
        representation, emotion_composition = sample["bot_representation"], sample["user_dialog_emotion_composition"]
        labels = f.one_hot(torch.cat(sample["bot_emotion"]), 7).to(dtype=torch.float)

        optimizer.zero_grad()

        output = model.representation_evolute(representation, emotion_composition)

        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for sample in tqdm(validation_dataloader, colour="blue"):
            representation, emotion_composition = sample["bot_representation"], sample[
                "user_dialog_emotion_composition"]
            labels = f.one_hot(torch.cat(sample["bot_emotion"]), 7).to(dtype=torch.float)

            output = model.representation_evolute(representation, emotion_composition)

            loss = loss_function(output, labels)


wandb.finish()
