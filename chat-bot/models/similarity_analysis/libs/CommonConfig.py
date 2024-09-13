import os
import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Union

import huggingface_hub
import numpy as np
import torch.cuda
import wandb
from dotenv import load_dotenv
from transformers.hf_argparser import HfArg

from .CommonUtils import value_candidate_check


@dataclass
class CommonScriptArguments:
    huggingface_api_token: Optional[str] = (
        HfArg(aliases=["--huggingface-api-token", "--huggingface-token", "--hf-token"], default=None))
    wandb_api_token: Optional[str] = (
        HfArg(aliases=["--wandb-api-token", "--wandb-token"], default=None))

    def __post_init__(self):
        load_dotenv(encoding="utf-8")

        if self.huggingface_api_token is None:
            self.huggingface_api_token = os.environ.get("HF_TOKEN", "")

        if self.wandb_api_token is None:
            self.wandb_api_token = os.environ.get("WANDB_API_KEY", "")

        huggingface_hub.login(token=self.huggingface_api_token, add_to_git_credential=True)
        wandb.login(key=self.wandb_api_token, relogin=True)

        torch.backends.cudnn.deterministic = True
        random.seed(hash("setting random seeds") % 2 ** 32 - 1)
        np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
        torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
        torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)


@dataclass
class CommonWanDBArguments:
    name: Optional[str] = HfArg(aliases=["--wandb-name"], default=None)
    job_type: Optional[str] = HfArg(aliases=["--wandb-job-type", "--job-type"], default=None)
    config: Union[Dict, str, None] = HfArg(aliases="--wandb-type", default_factory=dict)
    project: Optional[str] = HfArg(aliases="--wandb-project", default=None)
    group: Optional[str] = HfArg(aliases=["--wandb-group", "--group"], default=None)
    notes: Optional[str] = HfArg(aliases=["--wandb-notes", "--notes"], default=None)
    mode: Optional[Union[Literal["online", "offline", "disabled"], None]] = HfArg(aliases="--wandb-mode", default=None)
    allow_val_change: Optional[bool] = HfArg(aliases="--allow-val-change", default=False)
    resume: Optional[str] = HfArg(aliases="--wandb-resume", default=None)

    def __post_init__(self):
        module: List = ["Sentiment Analysis",
                        "Candidate Generator",
                        "Emotion Predictor",
                        "Emotion Model",
                        "Similarity Analysis",
                        "Response Generator"]

        self.group = value_candidate_check(self.group,
                                           use_default_value=True,
                                           default_value="",
                                           possible_values=module)
