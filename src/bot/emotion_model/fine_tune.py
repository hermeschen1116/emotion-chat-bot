from dataclasses import dataclass
from typing import Any, Optional

import torch
import wandb
from datasets import load_from_disk
from transformers.hf_argparser import HfArgumentParser, HfArg

from libs.CommonConfig import CommonWanDBArguments, CommonScriptArguments, get_torch_device


@dataclass
class ScriptArguments(CommonScriptArguments):
    dataset: Optional[str] = HfArg(aliases="--dataset", default="daily_dialog_for_RG_train:latest")
    dtype: Optional[Any] = HfArg(aliases="--dtype", default=torch.float32)
    device: Optional[str] = HfArg(aliases="--device", default_factory=get_torch_device)


parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments))
args, wandb_args, remain_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)

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

# Load Dataset
dataset_path = run.use_artifact(args.dataset).download()
dataset = load_from_disk(dataset_path)
