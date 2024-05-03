from .Attention import *

import torch


def representation_evolute(model, representation_src: list, emotion_compositions: list) -> torch.tensor:
    representations: list = representation_src
    for composition in emotion_compositions:
        new_representation: torch.tensor = model.forward(torch.tensor(representations[-1]), torch.tensor(composition))
        representations.append(list(new_representation))

    return torch.tensor(representations[1:], dtype=torch.float, requires_grad=True)


class EmotionModel(torch.nn.Module):
    def __init__(
        self,
        attention: str,
        dropout: Optional[float] = 0.5,
        scaler: Optional[float] = None,
        bias: Optional[bool] = True,
        dtype: Optional[Any] = torch.float,
    ):
        super(EmotionModel, self).__init__()

        self.__dtype: Any = dtype

        match attention:
            case "dot_product":
                self.__attention = DotProductAttention(dtype=dtype)
            case "scaled_dot_product":
                self.__attention = ScaledDotProductAttention(scaler, dtype=dtype)
            case "additive":
                self.__attention = AdditiveAttention(dropout, dtype=dtype)
            case "dual_linear":
                self.__attention = DualLinearAttention(dropout, dtype=dtype)

        self.__weight = torch.nn.LazyLinear(7, bias=bias, dtype=dtype)

    def forward(self, representation: torch.tensor, input_emotion: torch.tensor) -> torch.tensor:
        decomposed_representation: torch.tensor = representation.diag()

        attention_score: torch.tensor = self.__attention(input_emotion, decomposed_representation)

        difference: torch.tensor = torch.clamp(
            torch.sum(self.__weight((attention_score**3).to(dtype=self.__dtype)), dim=1), -1, 1)

        return representation + difference
