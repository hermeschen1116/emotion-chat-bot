from typing import Any, Optional

import torch

from attention.DotProductAttention import DotProductAttention


class EmotionModel:
    def __init__(self,
                 dropout: Optional[float] = 0.5,
                 scaler: Optional[float] = None,
                 bias: Optional[bool] = True,
                 dtype: Optional[Any] = torch.float32,
                 device: Optional[str] = "cpu") -> None:

        self.__model = DotProductAttention(dropout=dropout, dtype=dtype, device=device)

        self.__dtype: Any = dtype
        self.__device: str = device

        self.__weight = torch.nn.LazyLinear(7, bias=bias, device=device, dtype=dtype)

    def forward(self, representation: torch.tensor, input_emotion: torch.tensor) -> torch.tensor:
        attention_score: torch.tensor = (self.__model.forward(input_emotion, representation)
                                             .to(dtype=self.__dtype, device=self.__device))

        new_representation: torch.tensor = torch.clamp(self.__weight(attention_score), -1, 1)

        return new_representation
