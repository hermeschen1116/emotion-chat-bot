from typing import Any, Optional

import torch


class DualLinearAttention(torch.nn.Module):
    def __init__(self,
                 dropout: Optional[float] = None,
                 dtype: Optional[Any] = torch.float32,
                 device: Optional[str] = "cpu") -> None:
        super(DualLinearAttention, self).__init__()

        self.__dtype: Any = dtype
        self.__device: str = device

        self.__weight_Q = torch.nn.LazyLinear(7, bias=False, device=device, dtype=dtype)
        self.__weight_K = torch.nn.LazyLinear(7, bias=False, device=device, dtype=dtype)
        self.__dropout = torch.nn.Dropout(p=dropout if dropout is not None else 0.5)

    def forward(self, query: torch.tensor, keys: torch.tensor) -> torch.tensor:
        q: torch.tensor = self.__weight_Q(query)
        k: torch.tensor = self.__weight_K(keys)

        attention_score: torch.tensor = torch.sum(q * k, dim=1, dtype=self.__dtype)

        return self.__dropout(torch.softmax(attention_score, dim=0, dtype=self.__dtype))
