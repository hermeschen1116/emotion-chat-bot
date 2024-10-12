import random
from typing import Dict, List, Optional, Union

import numpy
import torch
from huggingface_hub.hub_mixin import PyTorchModelHubMixin
from torch import Tensor

from emotion_chat_bot.model.emotion_model.Attention import DotProductAttention

emotions: list = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]


def generate_dummy_representation(target_emotion: int) -> Tensor:
	weight_choice: numpy.ndarray = numpy.array([-1, 1])
	while True:
		dummy: Tensor = torch.rand(7, dtype=torch.float32)
		weight: Tensor = torch.tensor(numpy.random.choice(weight_choice, (7,)), dtype=torch.float32)
		dummy = torch.clamp(dummy * weight, -1, 1)
		if torch.argmax(dummy) == target_emotion:
			return dummy


def generate_representation(emotion_tendency: Optional[Union[int, Dict[str, float]]]) -> Tensor:
	if emotion_tendency is None:
		return generate_dummy_representation(random.randint(0, 6))
	if isinstance(emotion_tendency, int):
		return generate_dummy_representation(emotion_tendency)

	return torch.tensor([emotion_tendency[emotion] for emotion in emotions]).clamp(-1, 1)


def get_emotion_composition(analysis_result: list) -> Tensor:
	sentiment_composition: list = [result["score"] for result in analysis_result]

	return torch.tensor(sentiment_composition, dtype=torch.float32).softmax(dim=-1)


def representation_evolute(
	model, bot_emotion_representations: List[Tensor], user_emotion_compositions: List[Tensor]
) -> List[Tensor]:
	evolute_representations: list = bot_emotion_representations
	for composition in user_emotion_compositions:
		new_representation: Tensor = model.forward(evolute_representations[-1], composition)
		evolute_representations.append(new_representation)

	return evolute_representations


# for sweep
# def initialize_attention(attention: str, bias: bool = True, dtype: torch.dtype = torch.float) -> torch.nn.Module:
# 	match attention:
# 		case "dot_product":
# 			return DotProductAttention(dtype=dtype)
# 		case "scaled_dot_product":
# 			return ScaledDotProductAttention(dtype=dtype)
# 		case "additive":
# 			return AdditiveAttention(bias=bias, dtype=dtype)
# 		case "dual_linear":
# 			return DualLinearAttention(bias=bias, dtype=dtype)


class EmotionModel(torch.nn.Module, PyTorchModelHubMixin):
	def __init__(
		self, dropout: float = 0.32811879682394585, bias: bool = False, dtype: torch.dtype = torch.float32
	) -> None:
		super(EmotionModel, self).__init__()

		self.dtype: torch.dtype = dtype
		self.__attention: torch.nn.Module = DotProductAttention(dtype)
		self.__dropout = torch.nn.Dropout(p=dropout)
		self.__weight_D = torch.nn.Linear(7, 7, bias=bias, dtype=dtype)

	def forward(self, representation: Tensor, input_emotion: Tensor) -> Tensor:
		representation = representation.to(dtype=self.dtype)
		input_emotion = input_emotion.to(dtype=self.dtype)

		raw_attention: Tensor = self.__attention.forward(input_emotion, representation.squeeze().diag())

		attention_score: Tensor = raw_attention.softmax(-1).squeeze().diag()

		difference: Tensor = torch.clamp(torch.diagonal(self.__weight_D((attention_score**3))), -1, 1)

		return representation + difference
