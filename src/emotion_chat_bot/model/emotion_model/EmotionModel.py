from typing import Optional

import torch
from huggingface_hub.hub_mixin import PyTorchModelHubMixin


class EmotionModel(torch.nn.Module, PyTorchModelHubMixin):
	def __init__(
		self,
		input_size: int,
		hidden_size: int,
		device: str,
		dtype: torch.dtype,
		bias: bool = True,
		dropout: float = 0.0,
	) -> None:
		super(EmotionModel, self).__init__()

		self.hidden_size: int = hidden_size

		self.input_reset_gate = torch.nn.Linear(input_size, hidden_size, bias, device, dtype)
		self.input_update_gate = torch.nn.Linear(input_size, hidden_size, bias, device, dtype)
		self.input_new_gate = torch.nn.Linear(input_size, hidden_size, bias, device, dtype)
		self.hidden_reset_gate = torch.nn.Linear(hidden_size, hidden_size, bias, device, dtype)
		self.hidden_update_gate = torch.nn.Linear(hidden_size, hidden_size, bias, device, dtype)
		self.hidden_new_gate = torch.nn.Linear(hidden_size, hidden_size, bias, device, dtype)

	def forward(self, input: torch.Tensor, hidden_state: Optional[torch.Tensor]) -> torch.Tensor:
		if hidden_state is None:
			hidden_state = input.new_zeros(input.size(0), self.hidden_size)

		input_reset_gate_output: torch.Tensor = self.input_reset_gate(input)
		input_update_gate_output: torch.Tensor = self.input_update_gate(input)
		input_new_gate_output: torch.Tensor = self.input_new_gate(input)

		hidden_reset_gate_output: torch.Tensor = self.hidden_reset_gate(input)
		hidden_update_gate_output: torch.Tensor = self.hidden_update_gate(input)
		hidden_new_gate_output: torch.Tensor = self.hidden_new_gate(input)

		reset_gate_output = torch.sigmoid(input_reset_gate_output + hidden_reset_gate_output)
		update_gate_output = torch.sigmoid(input_update_gate_output + hidden_update_gate_output)
		new_gate_output = torch.sigmoid(input_new_gate_output + (reset_gate_output * hidden_new_gate_output))

		return (1 - update_gate_output) * hidden_state + update_gate_output * new_gate_output
