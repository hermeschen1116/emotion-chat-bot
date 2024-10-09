import os
from typing import Any, Dict, List, Optional

import huggingface_hub
import torch
import wandb
from dotenv import load_dotenv
from torch import Tensor
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score


def login_to_service() -> None:
	load_dotenv(encoding="utf-8")

	huggingface_hub.login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=True)
	wandb.login(key=os.environ.get("WANDB_API_KEY", ""), relogin=True)


def value_candidate_check(
	input_value: Any, possible_values: List[Any], use_default_value: bool, default_value: Optional[Any]
) -> Any:
	if input_value not in possible_values:
		error_message: str = (
			f"This parameter should be any of {', '.join(possible_values)}, your input is {input_value}"
		)
		if use_default_value:
			print(error_message)
			return default_value
		raise ValueError(error_message)

	return input_value


def get_torch_device() -> str:
	if torch.cuda.is_available():
		return "cuda"
	if torch.backends.mps.is_available():
		return "mps"

	return "cpu"


def calculate_evaluation_result(predictions: Tensor, truths: Tensor) -> Dict[str, Tensor]:
	accuracy: Tensor = multiclass_accuracy(predictions, truths, num_classes=7)
	f1_score: Tensor = multiclass_f1_score(predictions, truths, num_classes=7, average="weighted")

	return {"accuracy": accuracy, "f1_score": f1_score}
