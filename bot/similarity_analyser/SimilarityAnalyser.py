import torch
from torch import tensor, float32
from typing import Optional, Any, Annotated, Union
from dataclasses import dataclass


@dataclass
class ValueRange:
    min_value: float
    max_value: float

    def validate(self, x: float) -> float:
        if not (self.min_value <= x <= self.max_value):
            raise ValueError(f"{x} must be in range [{self.min_value}, {self.max_value}]")
        return x


class SimilarityAnalyser:
    def __init__(self,
                 threshold: Annotated[float, ValueRange(-1, 1)],
                 dtype: Optional[Any] = float32,
                 device: Optional[str] = "cpu") -> None:
        self.threshold: float = ValueRange(-1, 1).validate(threshold)
        self.dtype: Any = dtype
        self.device: str = device
        self.cache: dict = {}

    def __call__(self,
                 representations: Union[list[tensor], tensor],
                 ideal_representation: tensor,
                 eps: Optional[float] = 1e-08
                 ) -> tensor:
        self.cache["representations"]: tensor = representations if type(representations) is tensor \
                                                  else torch.stack(representations)

        self.cache["representations"] = (self.cache["representations"]
                                             .clone()
                                             .detach()
                                             .to(dtype=self.dtype, device=self.device))
        self.cache["ideal_representation"]: tensor = (ideal_representation.clone()
                                                                          .detach()
                                                                          .to(dtype=self.dtype, device=self.device))

        cosine_similarity: tensor = torch.cosine_similarity(self.cache["representations"],
                                                            self.cache["ideal_representation"],
                                                            eps=eps)
        length_of_representations: tensor = torch.norm(self.cache["representations"], dim=1, dtype=self.dtype)
        length_of_ideal_representation: tensor = torch.norm(self.cache["ideal_representation"], dtype=self.dtype)

        ratio_of_representations: tensor = length_of_representations / length_of_ideal_representation
        self.cache["similarity"]: tensor = cosine_similarity * ratio_of_representations

        return self.cache["similarity"]

    def get_most_similar_representation(self) -> dict[tensor, float]:
        mask: tensor = self.cache["similarity"] <= self.threshold
        valid_similarity_indices: tensor = torch.nonzero(mask)
        valid_similarity: tensor = self.cache["similarity"][valid_similarity_indices]
        max_similarity: float = torch.max(valid_similarity).unique().item()

        mask = self.cache["similarity"] == max_similarity
        representation_with_max_similarity_indices: tensor = torch.nonzero(mask)
        representations_with_max_similarity: list = (self.cache["representations"][representation_with_max_similarity_indices]
                                                         .squeeze()
                                                         .tolist())

        return {
            "representations": representations_with_max_similarity,
            "similarity": max_similarity
        }

    def get_most_similar_representation_index(self) -> int:
        max_similarity_value: float = self.get_most_similar_representation()["similarity"]
        mask: tensor = self.cache["similarity"] == max_similarity_value

        return torch.nonzero(mask).squeeze().item()
