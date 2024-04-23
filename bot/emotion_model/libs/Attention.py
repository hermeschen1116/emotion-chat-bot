from typing import Any, Optional

import torch


class DotProductAttention(torch.nn.Module):
    def __init__(self,
                 dtype: Optional[Any] = torch.float32,
                 device: Optional[str] = "cpu") -> None:
        super(DotProductAttention, self).__init__()

        self.__dtype: Any = dtype
        self.__device: str = device

    def forward(self, query: torch.tensor, keys: torch.tensor) -> torch.tensor:
        return (query * keys).to(dtype=self.__dtype, device=self.__device)


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self,
                 scaler: Optional[float] = None,
                 dtype: Optional[Any] = torch.float32,
                 device: Optional[str] = "cpu") -> None:
        super(ScaledDotProductAttention, self).__init__()

        self.__scaler: float = scaler
        self.__dtype: Any = dtype
        self.__device: str = device

        # self.__dropout = torch.nn.Dropout(p=dropout if dropout is not None else 0.5)

    def forward(self, query: torch.tensor, keys: torch.tensor) -> torch.tensor:
        self.__scaler = float(query.shape[-1]) if self.__scaler is None else self.__scaler

        return (query * keys / torch.sqrt_(torch.tensor(self.__scaler))).to(dtype=self.__dtype, device=self.__device)


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

        return (self.__weight_V(self.__dropout(q) + self.__dropout(k))).to(dtype=self.__dtype, device=self.__device)


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

        return (self.__dropout(q) * self.__dropout(k)).to(dtype=self.__dtype, device=self.__device)
