from typing import Any, Optional

import torch


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self,
                 dropout: Optional[float] = None,
                 scaler: Optional[float] = None,
                 dtype: Optional[Any] = torch.float32,
                 device: Optional[str] = "cpu") -> None:
        super(ScaledDotProductAttention, self).__init__()

        self.__scaler: float = scaler
        self.__dtype: Any = dtype
        self.__device: str = device

        self.__dropout = torch.nn.Dropout(p=dropout if dropout is not None else 0.5)

    def forward(self, query: torch.tensor, keys: torch.tensor) -> torch.tensor:
        self.__scaler = float(query.shape[-1]) if self.__scaler is None else self.__scaler

        attention_score: torch.tensor = torch.sum(query * keys / torch.sqrt_(torch.tensor(self.__scaler)),
                                                  dim=1,
                                                  dtype=self.__dtype)

        return self.__dropout(torch.softmax(attention_score, dim=0, dtype=self.__dtype))
