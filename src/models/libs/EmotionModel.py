import torch
from torch import Tensor

from .Attention import diagonal_softmax


def representation_evolute(
    model, representation_src: list, emotion_compositions: list
) -> Tensor:
    representations: list = representation_src
    for composition in emotion_compositions:
        new_representation: Tensor = model.forward(
            torch.tensor(representations[-1]), torch.tensor(composition)
        )
        representations.append(list(new_representation))

    return torch.tensor(representations[1:], dtype=torch.float, requires_grad=True)


class EmotionModel(torch.nn.Module):
    def __init__(
        self,
        dropout: float = 0.5,
        bias: bool = True,
        dtype: torch.dtype = torch.float,
        device: str = "cpu",
    ):
        super(EmotionModel, self).__init__()

        self.__dtype: torch.dtype = dtype
        self.__device: str = device

        self.__weight_Q = torch.nn.Linear(7, 7, bias=False, dtype=dtype, device=device)
        self.__weight_K = torch.nn.Linear(7, 7, bias=False, dtype=dtype, device=device)
        self.__dropout = torch.nn.Dropout(p=dropout)
        self.__weight_D = torch.nn.Linear(7, 7, bias=bias, dtype=dtype, device=device)

    def forward(self, representation: Tensor, input_emotion: Tensor) -> Tensor:
        decomposed_representation: Tensor = representation.diag().to(
            dtype=self.__dtype, device=self.__device
        )

        q: Tensor = self.__weight_Q(
            input_emotion.to(dtype=self.__dtype, device=self.__device)
        )
        k: Tensor = self.__weight_K(decomposed_representation)

        raw_attention: Tensor = torch.sum(
            self.__dropout(q) * self.__dropout(k), dim=1, dtype=self.__dtype
        )

        attention_score: Tensor = diagonal_softmax(
            raw_attention.squeeze().diag(), dtype=self.__dtype
        ).to(device=self.__device)

        difference: Tensor = torch.clamp(
            torch.sum(self.__weight_D((attention_score**3)), dim=1), -1, 1
        )

        return representation.to(dtype=self.__dtype, device=self.__device) + difference
