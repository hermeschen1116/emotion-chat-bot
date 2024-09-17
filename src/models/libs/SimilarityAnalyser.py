from typing import Any, Callable, Optional, Union

import torch
from torch import Tensor, float32


class SimilarityAnalyser:
    def __init__(
        self,
        threshold: float = 0.5,
        dtype: Optional[Any] = float32,
        device: str = "cpu",
    ) -> None:
        if not (0 < threshold < 1):
            raise ValueError("threshold must between 0 and 1 (exclusive)")
        self.__threshold: float = threshold
        self.__dtype: Any = dtype
        self.__device: str = device
        self.__cached_representations: Optional[Tensor] = None
        self.__cached_ideal_representation: Optional[Tensor] = None
        self.__cached_similarity: Optional[Tensor] = None

    @property
    def threshold(self) -> float:
        return self.__threshold

    @threshold.setter
    def threshold(self, new_threshold: float = 0.5):
        if not (0 < new_threshold < 1):
            raise ValueError("threshold must between 0 and 1 (exclusive)")
        self.__threshold = new_threshold
        self.__call__(self.__cached_representations, self.__cached_ideal_representation)

    def __calculate_ratio_of_length_of_representation(self) -> Tensor:
        length_of_representations: Tensor = torch.norm(
            self.__cached_representations, dim=1
        )
        length_of_ideal_representation: Tensor = torch.norm(
            self.__cached_ideal_representation
        )

        return length_of_representations / length_of_ideal_representation

    def __call__(
        self, representations: Union[list[Tensor], Tensor], ideal_representation: Tensor
    ) -> Tensor:
        self.__cached_representations = (
            representations
            if type(representations) is Tensor
            else torch.stack(representations)
        )
        self.__cached_representations = (
            self.__cached_representations.clone()
            .detach()
            .to(dtype=self.__dtype, device=self.__device)
        )

        self.__cached_ideal_representation = (
            ideal_representation.clone()
            .detach()
            .to(dtype=self.__dtype, device=self.__device)
        )

        cosine_similarity: Tensor = torch.cosine_similarity(
            self.__cached_representations, self.__cached_ideal_representation
        )
        ratio_of_representations: (
            Tensor
        ) = self.__calculate_ratio_of_length_of_representation()
        self.__cached_similarity = torch.clamp(
            cosine_similarity * ratio_of_representations, 0, 1
        )

        return self.__cached_similarity

    @staticmethod
    def __get_indices_of_filtered_tensor(
        target_tensor: Tensor, filter_func: Callable[[Tensor], Tensor]
    ) -> Tensor:
        mask: Tensor = filter_func(target_tensor)

        return torch.nonzero(mask)

    def get_max_similarity(self) -> float:
        valid_similarity_indices: Tensor = self.__get_indices_of_filtered_tensor(
            self.__cached_similarity, lambda x: x <= self.__threshold
        )
        if len(valid_similarity_indices) == 0:
            return 0

        valid_similarity: Tensor = self.__cached_similarity[valid_similarity_indices]
        return torch.max(valid_similarity).unique().item()

    def get_representation_with_max_similarity(self, max_similarity: float) -> list:
        representation_with_max_similarity_indices: (
            Tensor
        ) = self.__get_indices_of_filtered_tensor(
            self.__cached_similarity, lambda x: x == max_similarity
        )

        return (
            self.__cached_representations[representation_with_max_similarity_indices]
            .squeeze()
            .tolist()
        )

    def get_most_similar_representation(self) -> dict[Tensor, float]:
        max_similarity: float = self.get_max_similarity()
        representations_with_max_similarity: (
            list
        ) = self.get_representation_with_max_similarity(max_similarity)

        return {
            "representations": representations_with_max_similarity,
            "similarity": max_similarity,
        }

    def get_most_similar_representation_index(self) -> int:
        max_similarity_value: float = self.get_most_similar_representation()[
            "similarity"
        ]

        return (
            self.__get_indices_of_filtered_tensor(
                self.__cached_similarity, lambda x: x == max_similarity_value
            )
            .squeeze()
            .tolist()
        )
