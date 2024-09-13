import tempfile
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as f
import wandb
from datasets import load_from_disk
from libs import (CommonScriptArguments, CommonWanDBArguments, EmotionModel,
                  get_torch_device, representation_evolute)
from pyarrow import Field
from safetensors.torch import save_model
from torch.utils.data import DataLoader
from torcheval.metrics.functional import (multiclass_accuracy,
                                          multiclass_f1_score)
from tqdm.auto import tqdm
from transformers.hf_argparser import HfArg, HfArgumentParser


@dataclass
class ScriptArguments(CommonScriptArguments):
    dtype: Field[Optional[str]] = HfArg(aliases="--dtype", default="torch.float32")
    device: Field[Optional[str]] = HfArg(aliases="--device", default_factory=get_torch_device)


config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)
dtype: torch.dtype = eval(args.dtype)

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
dataset_path = run.use_artifact(wandb.config["dataset"]).download()
dataset = load_from_disk(dataset_path)

model = EmotionModel(dropout=wandb.config["dropout"], bias=wandb.config["bias"], dtype=dtype, device=args.device)

loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=wandb.config["learning_rate"])

train_dataloader = DataLoader(dataset["train"])
validation_dataloader = DataLoader(dataset["validation"])
for i in range(wandb.config["num_epochs"]):
    running_loss: float = 0
    model.train()
    for sample in tqdm(train_dataloader, colour="green"):
        representation, emotion_composition = sample["bot_representation"], sample["user_dialog_emotion_composition"]
        labels = f.one_hot(torch.cat(sample["bot_emotion"]), 7).to(dtype=torch.float)

        optimizer.zero_grad()

        output = representation_evolute(model, representation, emotion_composition)

        loss = loss_function(output, labels)
        wandb.log({"train/loss": loss.item()})
        running_loss += loss.item()

        loss.backward()
        optimizer.step()

    if i + 1 == wandb.config["num_epochs"]:
        wandb.log({"train/train_loss": running_loss / len(train_dataloader)})

    running_loss = 0
    true_labels: list = []
    predicted_labels: list = []
    model.eval()
    with torch.no_grad():
        for sample in tqdm(validation_dataloader, colour="blue"):
            representation, emotion_composition = sample["bot_representation"], sample[
                "user_dialog_emotion_composition"]
            labels = f.one_hot(torch.cat(sample["bot_emotion"]), 7).to(dtype=torch.float)

            output = representation_evolute(model, representation, emotion_composition)

            loss = loss_function(output, labels)
            wandb.log({"val/loss": loss.item()})
            running_loss += loss.item()
            true_labels += sample["bot_emotion"]
            predicted_labels.append(torch.argmax(output, dim=1))

        wandb.log({"val/val_loss": running_loss / len(validation_dataloader)})
        wandb.log({
            "val/f1_score": multiclass_f1_score(torch.cat(true_labels), torch.cat(predicted_labels),
                                                num_classes=7, average="weighted"),
            "val/accuracy": multiclass_accuracy(torch.cat(true_labels), torch.cat(predicted_labels),
                                                num_classes=7)
        })

model_artifact = wandb.Artifact(wandb.config["trained_model_name"], type="model")

model = torch.compile(model)
with tempfile.TemporaryDirectory() as temp_dir:
    save_model(model, f"{temp_dir}/{wandb.config['trained_model_name']}.safetensors")
    model_artifact.add_file(f"{temp_dir}/{wandb.config['trained_model_name']}.safetensors")
    run.log_artifact(model_artifact)

wandb.finish()
