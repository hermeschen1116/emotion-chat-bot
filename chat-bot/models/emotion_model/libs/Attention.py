from typing import Optional

from torch import Tensor
import torch


def diagonal_indices(source: Tensor) -> Tensor:
    source_shape: torch.Size = source.shape
    if len(source_shape) != 2:
        raise ValueError(f"""The source tensor is in shape {tuple(source_shape)}),
                             you should input a 2-D matrix.""")
    if source_shape[0] != source_shape[1]:
        raise ValueError(f"""The source tensor is in shape {tuple(source_shape)},
                             you should input a square matrix.""")

    return torch.tensor([i for i in range(source_shape[0])])


def diagonal_softmax(source: Tensor, dtype: torch.dtype = torch.float) -> Tensor:
    diagonal: Tensor = diagonal_indices(source)

    softmax_diagonal: Tensor = torch.softmax(source[diagonal, diagonal], dim=0, dtype=dtype)

    dest: Tensor = source
    dest[diagonal, diagonal] = softmax_diagonal.to(dtype=dest.dtype)

    return dest


class DotProductAttention(torch.nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float) -> None:
        super(DotProductAttention, self).__init__()
        self.__dtype: torch.dtype = dtype

    def forward(self, query: Tensor, keys: Tensor) -> Tensor:
        raw_attention: Tensor = torch.sum(query * keys, dim=1)

        return diagonal_softmax(raw_attention.squeeze().diag(), dtype=self.__dtype)


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self,
                 scaler: Optional[float] = None,
                 dtype: torch.dtype = torch.float) -> None:
        super(ScaledDotProductAttention, self).__init__()

        self.__dtype: torch.dtype = dtype
        self.__scaler: Optional[float] = scaler

    def forward(self, query: Tensor, keys: Tensor) -> Tensor:
        self.__scaler = float(query.shape[-1]) if self.__scaler is None else self.__scaler

        raw_attention: Tensor = torch.sum(query * keys / torch.sqrt_(torch.tensor(self.__scaler)), dim=1)

        return diagonal_softmax(raw_attention.squeeze().diag(), dtype=self.__dtype)


class AdditiveAttention(torch.nn.Module):
    def __init__(self,
                 dropout: Optional[float] = None,
                 dtype: torch.dtype = torch.float) -> None:
        super(AdditiveAttention, self).__init__()

        self.__dtype: torch.dtype = dtype

        self.__weight_Q = torch.nn.Linear(7, 7, bias=False, dtype=dtype)
        self.__weight_K = torch.nn.Linear(7, 7, bias=False, dtype=dtype)
        self.__weight_V = torch.nn.Linear(7, 7, bias=False, dtype=dtype)
        self.__dropout = torch.nn.Dropout(p=dropout if dropout is not None else 0.5)

    def forward(self, query: Tensor, keys: Tensor) -> Tensor:
        q: Tensor = self.__weight_Q(query.to(dtype=self.__dtype))
        k: Tensor = self.__weight_K(keys.to(dtype=self.__dtype))

        v: Tensor = self.__weight_V(torch.tanh(self.__dropout(q) + self.__dropout(k)))
        raw_attention: Tensor = torch.sum(v, dim=1)

        return diagonal_softmax(raw_attention.squeeze().diag(), dtype=self.__dtype)


class DualLinearAttention(torch.nn.Module):
    def __init__(self,
                 dropout: Optional[float] = None,
                 dtype: torch.dtype = torch.float) -> None:
        super(DualLinearAttention, self).__init__()

        self.__dtype: torch.dtype = dtype

        self.__weight_Q = torch.nn.Linear(7, 7, bias=False, dtype=dtype)
        self.__weight_K = torch.nn.Linear(7, 7, bias=False, dtype=dtype)
        self.__dropout = torch.nn.Dropout(p=dropout if dropout is not None else 0.5)

    def forward(self, query: Tensor, keys: Tensor) -> Tensor:
        q: Tensor = self.__weight_Q(query.to(dtype=self.__dtype))
        k: Tensor = self.__weight_K(keys.to(dtype=self.__dtype))

        raw_attention: Tensor = torch.sum(self.__dropout(q) * self.__dropout(k), dim=1)

        return diagonal_softmax(raw_attention.squeeze().diag(), dtype=self.__dtype)
