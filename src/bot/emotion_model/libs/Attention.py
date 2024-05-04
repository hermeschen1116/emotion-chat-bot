from typing import Optional

import torch


def diagonal_indices(source: torch.tensor) -> torch.tensor:
    source_shape: torch.Size = source.shape
    if len(source_shape) != 2:
        raise ValueError(f"""The source tensor is in shape {tuple(source_shape)}), 
                             you should input a 2-D matrix.""")
    if source_shape[0] != source_shape[1]:
        raise ValueError(f"""The source tensor is in shape {tuple(source_shape)}, 
                             you should input a square matrix.""")

    return torch.tensor([i for i in range(source_shape[0])])


def diagonal_softmax(source: torch.tensor, dtype: Optional[torch.dtype] = torch.float) -> torch.tensor:
    diagonal: torch.tensor = diagonal_indices(source)

    softmax_diagonal: torch.tensor = torch.softmax(source[diagonal, diagonal], dim=0, dtype=dtype)

    dest: torch.tensor = source
    dest[diagonal, diagonal] = softmax_diagonal.to(dtype=dest.dtype)

    return dest


class DotProductAttention(torch.nn.Module):
    def __init__(self, dtype: Optional[torch.dtype] = torch.float) -> None:
        super(DotProductAttention, self).__init__()
        self.__dtype: torch.dtype = dtype

    def forward(self, query: torch.tensor, keys: torch.tensor) -> torch.tensor:
        raw_attention: torch.tensor = torch.sum(query * keys, dim=1)

        return diagonal_softmax(raw_attention.squeeze().diag(), dtype=self.__dtype)


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self,
                 scaler: Optional[float] = None,
                 dtype: torch.dtype = torch.float) -> None:
        super(ScaledDotProductAttention, self).__init__()

        self.__dtype: torch.dtype = dtype
        self.__scaler: float = scaler

    def forward(self, query: torch.tensor, keys: torch.tensor) -> torch.tensor:
        self.__scaler = float(query.shape[-1]) if self.__scaler is None else self.__scaler

        raw_attention: torch.tensor = torch.sum(query * keys / torch.sqrt_(torch.tensor(self.__scaler)), dim=1)

        return diagonal_softmax(raw_attention.squeeze().diag(), dtype=self.__dtype)


class AdditiveAttention(torch.nn.Module):
    def __init__(self,
                 dropout: Optional[float] = None,
                 dtype: Optional[torch.dtype] = torch.float) -> None:
        super(AdditiveAttention, self).__init__()

        self.__dtype: torch.dtype = dtype

        self.__weight_Q = torch.nn.Linear(7, 7, bias=False, dtype=dtype)
        self.__weight_K = torch.nn.Linear(7, 7, bias=False, dtype=dtype)
        self.__weight_V = torch.nn.Linear(7, 7, bias=False, dtype=dtype)
        self.__dropout = torch.nn.Dropout(p=dropout if dropout is not None else 0.5)

    def forward(self, query: torch.tensor, keys: torch.tensor) -> torch.tensor:
        q: torch.tensor = self.__weight_Q(query.to(dtype=self.__dtype))
        k: torch.tensor = self.__weight_K(keys.to(dtype=self.__dtype))

        v: torch.tensor = self.__weight_V(torch.tanh(self.__dropout(q) + self.__dropout(k)))
        raw_attention: torch.tensor = torch.sum(v, dim=1)

        return diagonal_softmax(raw_attention.squeeze().diag(), dtype=self.__dtype)


class DualLinearAttention(torch.nn.Module):
    def __init__(self,
                 dropout: Optional[float] = None,
                 dtype: Optional[torch.dtype] = torch.float) -> None:
        super(DualLinearAttention, self).__init__()

        self.__dtype: torch.dtype = dtype

        self.__weight_Q = torch.nn.Linear(7, 7, bias=False, dtype=dtype)
        self.__weight_K = torch.nn.Linear(7, 7, bias=False, dtype=dtype)
        self.__dropout = torch.nn.Dropout(p=dropout if dropout is not None else 0.5)

    def forward(self, query: torch.tensor, keys: torch.tensor) -> torch.tensor:
        q: torch.tensor = self.__weight_Q(query.to(dtype=self.__dtype))
        k: torch.tensor = self.__weight_K(keys.to(dtype=self.__dtype))

        raw_attention: torch.tensor = torch.sum(self.__dropout(q) * self.__dropout(k), dim=1)

        return diagonal_softmax(raw_attention.squeeze().diag(), dtype=self.__dtype)
