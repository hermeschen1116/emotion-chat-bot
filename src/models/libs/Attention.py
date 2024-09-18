import torch
from torch import Tensor


def diagonal_indices(source: Tensor) -> Tensor:
	source_shape: torch.Size = source.shape
	if len(source_shape) != 2:
		raise ValueError(
			f"""The source tensor is in shape {tuple(source_shape)}),
                             you should input a 2-D matrix."""
		)
	if source_shape[0] != source_shape[1]:
		raise ValueError(
			f"""The source tensor is in shape {tuple(source_shape)},
                             you should input a square matrix."""
		)

	return torch.tensor([i for i in range(source_shape[0])])


def diagonal_softmax(source: Tensor) -> Tensor:
	diagonal: Tensor = diagonal_indices(source)

	softmax_diagonal: Tensor = torch.softmax(source[diagonal, diagonal], dim=0)

	dest: Tensor = source
	dest[diagonal, diagonal] = softmax_diagonal

	return dest


class DotProductAttention(torch.nn.Module):
	def __init__(self, dtype: torch.dtype = torch.float, device: str = "cpu") -> None:
		super(DotProductAttention, self).__init__()
		self.__dtype: torch.dtype = dtype
		self.__device: str = device

	def forward(self, query: Tensor, keys: Tensor) -> Tensor:
		query = query.to(dtype=self.__dtype, device=self.__device)
		keys = keys.to(dtype=self.__dtype, device=self.__device)

		raw_attention: Tensor = torch.sum(query * keys, dim=1)

		return diagonal_softmax(raw_attention.diag())


class ScaledDotProductAttention(torch.nn.Module):
	def __init__(self, dtype: torch.dtype = torch.float, device: str = "cpu") -> None:
		super(ScaledDotProductAttention, self).__init__()

		self.__dtype: torch.dtype = dtype
		self.__device: str = device

	def forward(self, query: Tensor, keys: Tensor) -> Tensor:
		query = query.to(dtype=self.__dtype, device=self.__device)
		keys = keys.to(dtype=self.__dtype, device=self.__device)

		scaler: Tensor = torch.tensor(query.shape[-1])

		raw_attention: Tensor = torch.sum(query * keys / torch.sqrt_(scaler), dim=1)

		return diagonal_softmax(raw_attention.diag())


class AdditiveAttention(torch.nn.Module):
	def __init__(self, bias: bool = True, dtype: torch.dtype = torch.float, device: str = "cpu") -> None:
		super(AdditiveAttention, self).__init__()

		self.__dtype: torch.dtype = dtype
		self.__device: str = device

		self.__weight_Q = torch.nn.Linear(7, 7, bias=bias, dtype=dtype, device=device)
		self.__weight_K = torch.nn.Linear(7, 7, bias=bias, dtype=dtype, device=device)
		self.__weight_V = torch.nn.Linear(7, 7, bias=bias, dtype=dtype, device=device)

	def forward(self, query: Tensor, keys: Tensor) -> Tensor:
		query = query.to(dtype=self.__dtype, device=self.__device)
		keys = keys.to(dtype=self.__dtype, device=self.__device)

		q: Tensor = self.__weight_Q(query)
		k: Tensor = self.__weight_K(keys)

		v: Tensor = self.__weight_V(torch.tanh(q + k))
		raw_attention: Tensor = torch.sum(v, dim=1)

		return diagonal_softmax(raw_attention.diag())


class DualLinearAttention(torch.nn.Module):
	def __init__(self, bias: bool = True, dtype: torch.dtype = torch.float, device: str = "cpu") -> None:
		super(DualLinearAttention, self).__init__()

		self.__dtype: torch.dtype = dtype
		self.__device: str = device

		self.__weight_Q = torch.nn.Linear(7, 7, bias=bias, dtype=dtype, device=device)
		self.__weight_K = torch.nn.Linear(7, 7, bias=bias, dtype=dtype, device=device)

	def forward(self, query: Tensor, keys: Tensor) -> Tensor:
		query = query.to(dtype=self.__dtype, device=self.__device)
		keys = keys.to(dtype=self.__dtype, device=self.__device)

		q: Tensor = self.__weight_Q(query)
		k: Tensor = self.__weight_K(keys)

		raw_attention: Tensor = torch.sum(q * k, dim=1)

		return diagonal_softmax(raw_attention.diag())
