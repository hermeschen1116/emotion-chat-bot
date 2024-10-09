import random
from dataclasses import Field, dataclass
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import torch.cuda
from transformers.hf_argparser import HfArg

from emotion_chat_bot.utils.CommonUtils import login_to_service, value_candidate_check


@dataclass
class CommonScriptArguments:
	@staticmethod
	def __post_init__():
		login_to_service()

		torch.backends.cudnn.deterministic = True
		random.seed(hash("setting random seeds") % 2**32 - 1)
		np.random.seed(hash("improves reproducibility") % 2**32 - 1)
		torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
		torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)


@dataclass
class CommonWanDBArguments:
	name: Field[Optional[str]] = HfArg(aliases=["--wandb-name"], default=None)
	job_type: Field[Optional[str]] = HfArg(aliases=["--wandb-job-type", "--job-type"], default=None)
	config: Field[Union[Dict, str, None]] = HfArg(aliases="--wandb-type", default_factory=dict)
	project: Field[Optional[str]] = HfArg(aliases="--wandb-project", default=None)
	group: Field[
		Optional[
			Literal[
				"Sentiment Analysis",
				"Emotion Predictor",
				"Emotion Model",
				"Similarity Analysis",
				"Response Generator",
			]
		]
	] = HfArg(aliases=["--wandb-group", "--group"], default=None)
	notes: Field[Optional[str]] = HfArg(aliases=["--wandb-notes", "--notes"], default=None)
	mode: Field[Optional[Literal["online", "offline", "disabled"]]] = HfArg(aliases="--wandb-mode", default=None)
	allow_val_change: Field[Optional[bool]] = HfArg(aliases="--allow-val-change", default=False)
	resume: Field[Optional[str]] = HfArg(aliases="--wandb-resume", default=None)

	def __post_init__(self):
		module: List = [
			"Sentiment Analysis",
			"Emotion Predictor",
			"Emotion Model",
			"Similarity Analysis",
			"Response Generator",
		]

		self.group = value_candidate_check(self.group, use_default_value=True, default_value="", possible_values=module)
