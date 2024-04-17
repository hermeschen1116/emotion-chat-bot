from typing import Any, Optional

import torch


class DotProductAttention(torch.nn.Module):
    def __init__(self,
                 dropout: Optional[float] = None,
                 dtype: Optional[Any] = torch.float32,
                 device: Optional[str] = "cpu") -> None:
        super(DotProductAttention, self).__init__()

        self.__dtype: Any = dtype
        self.__device: str = device

    def forward(self, query: torch.tensor, keys: torch.tensor) -> torch.tensor:
        attention_score: torch.tensor = torch.sum(query * keys, dim=1, dtype=self.__dtype)
        return torch.softmax(attention_score, dim=0, dtype=self.__dtype)
