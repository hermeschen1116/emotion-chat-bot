from dataclasses import Field, dataclass
from typing import Dict, List, Literal, Optional, Union

from transformers.hf_argparser import HfArg

from emotion_chat_bot.utils.Helper import login_to_service, value_candidate_validate
from emotion_chat_bot.utils.Train import setup_reproducibility


@dataclass
class CommonScriptArguments:
	@staticmethod
	def __post_init__():
		login_to_service()
		setup_reproducibility()


@dataclass
class CommonWanDBArguments:
	name: Field[Optional[str]] = HfArg(aliases=["--wandb-name"], default=None)
	job_type: Field[Optional[str]] = HfArg(aliases=["--wandb-job-type", "--job-type"], default=None)
	config: Field[Union[Dict, str, None]] = HfArg(aliases="--wandb-type", default_factory=dict)
	project: Field[Optional[str]] = HfArg(aliases="--wandb-project", default=None)
	group: Field[
		Optional[Literal["Sentiment Analyzer", "Emotion Predictor", "Emotion Model", "Response Generator"]]
	] = HfArg(aliases=["--wandb-group", "--group"], default=None)
	notes: Field[Optional[str]] = HfArg(aliases=["--wandb-notes", "--notes"], default=None)
	mode: Field[Optional[Literal["online", "offline", "disabled"]]] = HfArg(aliases="--wandb-mode", default=None)
	resume: Field[Optional[str]] = HfArg(aliases="--wandb-resume", default=None)

	def __post_init__(self):
		module: List = ["Sentiment Analysis", "Emotion Predictor", "Emotion Model", "Response Generator"]
		mode: list = ["online", "offline", "disabled"]

		self.group = value_candidate_validate(
			self.group, use_default_value=True, default_value="", possible_values=module
		)

		self.mode = value_candidate_validate(self.mode, use_default_value=False, default_value="", possible_values=mode)
