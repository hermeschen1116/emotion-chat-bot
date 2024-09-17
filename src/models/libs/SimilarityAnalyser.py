from typing import Any, Optional, Union

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
