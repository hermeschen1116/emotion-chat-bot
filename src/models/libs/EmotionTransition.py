from typing import List, Optional

import torch
from huggingface_hub.hub_mixin import PyTorchModelHubMixin
from torch import Tensor

from .Attention import DotProductAttention


def generate_dummy_representation(target_emotion: int) -> Tensor:
	while True:
		dummy = torch.clamp(torch.rand(7, dtype=torch.float32), -1, 1)
		if torch.argmax(dummy) == target_emotion:
			return dummy


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
		self,
		dropout: float = 0.5321222767142184,
		bias: bool = False,
		dtype: torch.dtype = torch.bfloat16,
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


class EmotionPresentationSimilarityAnalyser:
	def __init__(self, ideal_emotion_representation: Optional[Tensor], threshold: float = 0.7) -> None:
		if not (0 < threshold < 1):
			raise ValueError("threshold must between 0 and 1 (exclusive)")
		self.__threshold: float = threshold
		self.__ideal_emotion_representation: Tensor = ideal_emotion_representation
		self.__length_of_ideal_emotion_representation: Tensor = None
		if ideal_emotion_representation is not None:
			self.__length_of_ideal_emotion_representation = torch.norm(ideal_emotion_representation)

	@property
	def threshold(self) -> float:
		return self.__threshold

	@property
	def ideal_emotion_representation(self) -> Tensor:
		return self.__ideal_emotion_representation

	@ideal_emotion_representation.setter
	def ideal_emotion_representation(self, ideal_emotion_representation) -> None:
		self.__ideal_emotion_representation = ideal_emotion_representation
		self.__length_of_ideal_emotion_representation = torch.norm(ideal_emotion_representation)

	def __call__(self, representations: Tensor) -> Tensor:
		if self.__ideal_emotion_representation is None:
			raise ValueError("ideal_emotion_representation is not set")

		length_of_representations: Tensor = torch.norm(representations, dim=1).to(representations.device)

		length_ratio_between_representations: Tensor = (
			length_of_representations / self.__length_of_ideal_emotion_representation
		)

		similarity_between_representations: Tensor = torch.cosine_similarity(
			representations, self.__ideal_emotion_representation.to(representations.device)
		)

		similarity_scores: Tensor = similarity_between_representations * length_ratio_between_representations
		similarity_scores = torch.clamp(similarity_scores, min=0, max=1)

		score_filter: Tensor = similarity_scores <= self.__threshold
		similarity_scores = similarity_scores * score_filter

		return similarity_scores
