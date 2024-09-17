import torch
from huggingface_hub.hub_mixin import PyTorchModelHubMixin
from torch import Tensor, nn

from .Attention import diagonal_softmax


def representation_evolute(model, representation_src: list, emotion_compositions: list) -> Tensor:
	representations: list = representation_src
	for composition in emotion_compositions:
		new_representation: Tensor = model.forward(torch.tensor(representations[-1]), torch.tensor(composition))
		representations.append(list(new_representation))

	return torch.tensor(representations[1:], dtype=torch.float, requires_grad=True)


class EmotionModel(torch.nn.Module, PyTorchModelHubMixin):
	def __init__(
		self,
		dropout: float = 0.5,
		bias: bool = True,
		dtype: torch.dtype = torch.float32,
	):
		super(EmotionModel, self).__init__()

		self.device = nn.Parameter(torch.empty(0)).device
		self.dtype = dtype
		self.__weight_Q = torch.nn.Linear(7, 7, bias=False, device=self.device, dtype=self.dtype)
		self.__weight_K = torch.nn.Linear(7, 7, bias=False, device=self.device, dtype=self.dtype)
		self.__dropout = torch.nn.Dropout(p=dropout)
		self.__weight_D = torch.nn.Linear(7, 7, bias=bias, device=self.device, dtype=self.dtype)

	def forward(self, representation: Tensor, input_emotion: Tensor) -> Tensor:
		representation = representation.to(device=self.device, dtype=self.dtype)
		input_emotion = input_emotion.to(device=self.device, dtype=self.dtype)

		decomposed_representation: Tensor = representation.diag()

		q: Tensor = self.__weight_Q(input_emotion)
		k: Tensor = self.__weight_K(decomposed_representation)

		raw_attention: Tensor = torch.sum(self.__dropout(q) * self.__dropout(k), dim=1)

		attention_score: Tensor = diagonal_softmax(raw_attention.squeeze().diag())

		difference: Tensor = torch.clamp(torch.sum(self.__weight_D((attention_score**3)), dim=1), -1, 1)

		return representation + difference
