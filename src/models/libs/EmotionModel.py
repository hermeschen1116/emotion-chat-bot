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


def representation_evolute(model, representation_src: list, emotion_compositions: list) -> Tensor:
	representations: list = representation_src
	for composition in emotion_compositions:
		new_representation: Tensor = model.forward(torch.tensor(representations[-1]), torch.tensor(composition))
		representations.append(list(new_representation))

	return torch.tensor(representations[1:], dtype=torch.float, requires_grad=True)


def initialize_attention(attention: str, **kwargs) -> torch.nn.Module:
	match attention:
		case "dot_product":
			return DotProductAttention(dtype=kwargs.dtype, device=kwargs.device)
		case "scaled_dot_product":
			return ScaledDotProductAttention(dtype=kwargs.dtype, device=kwargs.device)
		case "additive":
			return AdditiveAttention(bias=kwargs.bias, dtype=kwargs.dtype, device=kwargs.device)
		case "dual_linear":
			return DualLinearAttention(bias=kwargs.bias, dtype=kwargs.dtype, device=kwargs.device)


class EmotionModel(torch.nn.Module, PyTorchModelHubMixin):
	def __init__(
		self,
		attention: str,
		dropout: float = 0.5,
		bias: bool = True,
		dtype: torch.dtype = torch.float32,
		device: str = "cpu",
	):
		super(EmotionModel, self).__init__()

		self.device: str = device
		self.dtype: torch.dtype = dtype
		self.__attention: torch.nn.Module = initialize_attention(attention, bias=bias, device=device, dtype=dtype)
		self.__dropout = torch.nn.Dropout(p=dropout)
		self.__weight_D = torch.nn.Linear(7, 7, bias=bias, device=self.device, dtype=self.dtype)

	def forward(self, representation: Tensor, input_emotion: Tensor) -> Tensor:
		representation = representation.to(device=self.device, dtype=self.dtype)
		input_emotion = input_emotion.to(device=self.device, dtype=self.dtype)

		decomposed_representation: Tensor = representation.diag()

		raw_attention: Tensor = self.__attention.forward(input_emotion, decomposed_representation)

		attention_score: Tensor = diagonal_softmax(raw_attention.squeeze().diag())

		difference: Tensor = torch.clamp(torch.sum(self.__weight_D((attention_score**3)), dim=1), -1, 1)

		return representation + difference
