import os
import random
from dataclasses import Field, dataclass
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
	huggingface_api_token: Field[Optional[str]] = (
        HfArg(aliases=["--huggingface-api-token", "--huggingface-token", "--hf-token"], default=os.environ.get("HF_TOKEN", "")))
	wandb_api_token: Field[Optional[str]] = (
        HfArg(aliases=["--wandb-api-token", "--wandb-token"], default=os.environ.get("WANDB_API_KEY", "")))

	def __post_init__(self):
		load_dotenv(encoding="utf-8")

		if self.huggingface_api_token == "":
			raise ValueError("HF_TOKEN is empty")

		if self.wandb_api_token == "":
			raise ValueError("WADDB_API_KEY is empty")

		huggingface_hub.login(token=self.huggingface_api_token, add_to_git_credential=True)
		wandb.login(key=self.wandb_api_token, relogin=True)

		torch.backends.cudnn.deterministic = True
		random.seed(hash("setting random seeds") % 2 ** 32 - 1)
		np.random.seed(hash("improves reproducibility") % 2 ** 32 - 1)
		torch.manual_seed(hash("by removing stochasticity") % 2 ** 32 - 1)
		torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2 ** 32 - 1)


@dataclass
class CommonWanDBArguments:
    name: Field[Optional[str]] = HfArg(aliases=["--wandb-name"], default=None)
    job_type: Field[Optional[str]] = HfArg(aliases=["--wandb-job-type", "--job-type"], default=None)
    config: Field[Union[Dict, str, None]] = HfArg(aliases="--wandb-type", default_factory=dict)
    project: Field[Optional[str]] = HfArg(aliases="--wandb-project", default=None)
    group: Field[Optional[Union[Literal[
    				"Sentiment Analysis",
                    "Emotion Predictor",
                    "Emotion Model",
                    "Similarity Analysis",
                    "Response Generator"], None]]] = HfArg(aliases=["--wandb-group", "--group"], default=None)
    notes: Field[Optional[str]] = HfArg(aliases=["--wandb-notes", "--notes"], default=None)
    mode: Field[Optional[Union[Literal["online", "offline", "disabled"], None]]] = HfArg(aliases="--wandb-mode", default=None)
    allow_val_change: Field[Optional[bool]] = HfArg(aliases="--allow-val-change", default=False)
    resume: Field[Optional[str]] = HfArg(aliases="--wandb-resume", default=None)

    def __post_init__(self):
        module: List = ["Sentiment Analysis",
                        "Emotion Predictor",
                        "Emotion Model",
                        "Similarity Analysis",
                        "Response Generator"]

        self.group = value_candidate_check(self.group,
                                           use_default_value=True,
                                           default_value="",
                                           possible_values=module)
