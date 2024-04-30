from dataclasses import dataclass
from typing import Any, Optional

import torch
import wandb
from datasets import load_from_disk
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from transformers.hf_argparser import HfArgumentParser, HfArg
from torch.utils.data import DataLoader

from libs.CommonConfig import CommonWanDBArguments, CommonScriptArguments, get_torch_device
from libs.EmotionModel import EmotionModel


@dataclass
class ScriptArguments(CommonScriptArguments):
    dataset: Optional[str] = HfArg(aliases="--dataset", default="daily_dialog_for_EM:latest")
    dtype: Optional[Any] = HfArg(aliases="--dtype", default=torch.float32)
    device: Optional[str] = HfArg(aliases="--device", default_factory=get_torch_device)


parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file("./args/train_arg.json")

# Initialize Wandb
logger = WandbLogger(job_type=wandb_args.job_type,
                     config=wandb_args.config,
                     project=wandb_args.project,
                     group=wandb_args.group,
                     notes=wandb_args.notes,
                     mode=wandb_args.mode,
                     resume=wandb_args.resume)

# Load Dataset
dataset_path = logger.use_artifact(args.dataset).download()
dataset = load_from_disk(dataset_path)

model = EmotionModel(wandb.config["attention_type"])
model = torch.compile(model)

trainer = Trainer(logger=logger,
                  enable_progress_bar=True,
                  enable_model_summary=True,
                  max_epochs=wandb.config["num_epochs"],
                  log_every_n_steps=10,
                  enable_checkpointing=True)

train_dataloader = DataLoader(dataset["train"], num_workers=16)
validation_dataloader = DataLoader(dataset["validation"], num_workers=16)
for sample in train_dataloader:
    print(sample)
    break
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)
