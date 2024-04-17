from typing import Any, Optional

import torch


class AdditiveAttention(torch.nn.Module):
    def __init__(self,
                 dropout: Optional[float] = None,
                 dtype: Optional[Any] = torch.float32,
                 device: Optional[str] = "cpu") -> None:
        super(AdditiveAttention, self).__init__()

        self.__dtype: Any = dtype
        self.__device: str = device

        self.__weight_Q = torch.nn.LazyLinear(7, bias=False, device=device, dtype=dtype)
        self.__weight_K = torch.nn.LazyLinear(7, bias=False, device=device, dtype=dtype)
        self.__weight_V = torch.nn.LazyLinear(1, bias=False, device=device, dtype=dtype)
        self.__dropout = torch.nn.Dropout(p=dropout if dropout is not None else 0.5)

    def forward(self, query: torch.tensor, keys: torch.tensor) -> torch.tensor:
        q: torch.tensor = self.__weight_Q(query)
        k: torch.tensor = self.__weight_K(keys)

        f: torch.tensor = self.__weight_V(q + k)
        attention_score: torch.tensor = torch.sum(torch.tanh(f), dim=1, dtype=self.__dtype)

        return self.__dropout(torch.softmax(attention_score, dim=0, dtype=self.__dtype))
