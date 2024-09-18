from typing import List

import torch
from huggingface_hub.hub_mixin import PyTorchModelHubMixin
from torch import Tensor

from .Attention import (
	AdditiveAttention,
	DotProductAttention,
	DualLinearAttention,
	ScaledDotProductAttention,
	diagonal_softmax,
)


def representation_evolute(model, bot_emotion_representations: List[Tensor], user_emotion_compositions: List[Tensor]) -> List[Tensor]:
	evolute_representations: list = bot_emotion_representations
	for composition in user_emotion_compositions:
		new_representation: Tensor = model.forward(evolute_representations[-1], composition)
		evolute_representations.append(new_representation)

	return evolute_representations


def initialize_attention(
	attention: str, bias: bool = True, dtype: torch.dtype = torch.float
) -> torch.nn.Module:
	match attention:
		case "dot_product":
			return DotProductAttention(dtype=dtype)
		case "scaled_dot_product":
			return ScaledDotProductAttention(dtype=dtype)
		case "additive":
			return AdditiveAttention(bias=bias, dtype=dtype)
		case "dual_linear":
			return DualLinearAttention(bias=bias, dtype=dtype)


class EmotionModel(torch.nn.Module, PyTorchModelHubMixin):
	def __init__(
		self,
		attention: str,
		dropout: float = 0.5,
		bias: bool = True,
		dtype: torch.dtype = torch.float32,
	) -> None:
		super(EmotionModel, self).__init__()

		self.dtype: torch.dtype = dtype
		self.__attention: torch.nn.Module = initialize_attention(attention, bias, dtype)
		self.__dropout = torch.nn.Dropout(p=dropout)
		self.__weight_D = torch.nn.Linear(7, 7, bias=bias, dtype=dtype)

	def forward(self, representation: Tensor, input_emotion: Tensor) -> Tensor:
		representation = representation.to(dtype=self.dtype)
		input_emotion = input_emotion.to(dtype=self.dtype)

		raw_attention: Tensor = self.__attention.forward(input_emotion, representation.squeeze().diag())

		attention_score: Tensor = raw_attention.softmax(-1).squeeze().diag()

		difference: Tensor = torch.clamp(torch.diagonal(self.__weight_D((attention_score**3))), -1, 1)

		return representation + difference
