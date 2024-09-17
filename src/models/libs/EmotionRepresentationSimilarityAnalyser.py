from typing import Optional

import torch
from torch import Tensor


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
