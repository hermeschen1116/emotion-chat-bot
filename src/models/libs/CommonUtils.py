from typing import Any, List, Optional

import torch


def value_candidate_check(
    input_value: Any,
    possible_values: List[Any],
    use_default_value: bool,
    default_value: Optional[Any],
) -> Any:
    if input_value not in possible_values:
        error_message: str = f"This parameter should be any of {', '.join(possible_values)}, your input is {input_value}"
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
