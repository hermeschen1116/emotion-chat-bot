import torch
from torch import Tensor


class DotProductAttention(torch.nn.Module):
	def __init__(self, dtype: torch.dtype = torch.float) -> None:
		super(DotProductAttention, self).__init__()
		self.__dtype: torch.dtype = dtype

	def forward(self, query: Tensor, keys: Tensor) -> Tensor:
		query = query.to(dtype=self.__dtype)
		keys = keys.to(dtype=self.__dtype)

		return query.matmul(keys)


class ScaledDotProductAttention(torch.nn.Module):
	def __init__(self, dtype: torch.dtype = torch.float) -> None:
		super(ScaledDotProductAttention, self).__init__()

		self.__dtype: torch.dtype = dtype

	def forward(self, query: Tensor, keys: Tensor) -> Tensor:
		query = query.to(dtype=self.__dtype)
		keys = keys.to(dtype=self.__dtype)

		return query.matmul(keys) / torch.tensor(query.shape[-1]).sqrt_()


class AdditiveAttention(torch.nn.Module):
	def __init__(self, bias: bool = True, dtype: torch.dtype = torch.float) -> None:
		super(AdditiveAttention, self).__init__()

		self.__dtype: torch.dtype = dtype

		self.__weight_Q = torch.nn.Linear(7, 7, bias=bias, dtype=dtype)
		self.__weight_K = torch.nn.Linear(7, 7, bias=bias, dtype=dtype)
		self.__weight_V = torch.nn.Linear(7, 7, bias=bias, dtype=dtype)

	def forward(self, query: Tensor, keys: Tensor) -> Tensor:
		query = query.to(dtype=self.__dtype)
		keys = keys.to(dtype=self.__dtype)

		q: Tensor = self.__weight_Q(query)
		k: Tensor = self.__weight_K(keys)

		return torch.sum(self.__weight_V(torch.tanh(q + k)), dim=0)


class DualLinearAttention(torch.nn.Module):
	def __init__(self, bias: bool = True, dtype: torch.dtype = torch.float) -> None:
		super(DualLinearAttention, self).__init__()

		self.__dtype: torch.dtype = dtype

		self.__weight_Q = torch.nn.Linear(7, 7, bias=bias, dtype=dtype)
		self.__weight_K = torch.nn.Linear(7, 7, bias=bias, dtype=dtype)

	def forward(self, query: Tensor, keys: Tensor) -> Tensor:
		query = query.to(dtype=self.__dtype)
		keys = keys.to(dtype=self.__dtype)

		q: Tensor = self.__weight_Q(query)
		k: Tensor = self.__weight_K(keys)

		return q.matmul(k)
