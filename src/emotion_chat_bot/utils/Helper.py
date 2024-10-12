import os
from typing import Any, List, Optional

import huggingface_hub
import wandb
from dotenv import load_dotenv


def login_to_service() -> None:
	load_dotenv(encoding="utf-8")

	huggingface_hub.login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=True)
	wandb.login(key=os.environ.get("WANDB_API_KEY", ""), relogin=True)


def value_candidate_validate(
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
