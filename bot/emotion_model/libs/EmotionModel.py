import torch

from Attention import *


class EmotionModel(torch.nn.Module):
    def __init__(self,
                 attention: str,
                 dropout: Optional[float] = 0.5,
                 scaler: Optional[float] = None,
                 bias: Optional[bool] = True,
                 dtype: Optional[Any] = torch.float32,
                 device: Optional[str] = "cpu") -> None:
        super(EmotionModel, self).__init__()

        match attention:
            case "dot_product":
                self.__model = DotProductAttention(dtype=dtype, device=device)
            case "scaled_dot_product":
                self.__model = ScaledDotProductAttention(scaler, dtype=dtype, device=device)
            case "additive":
                self.__model = AdditiveAttention(dropout, dtype=dtype, device=device)
            case "dual_linear":
                self.__model = DualLinearAttention(dropout, dtype=dtype, device=device)

        self.__dtype: Any = dtype
        self.__device: str = device

        self.__weight = torch.nn.LazyLinear(7, bias=bias, device=device, dtype=dtype)

    def forward(self, representation: torch.tensor, input_emotion: torch.tensor) -> torch.tensor:
        output: torch.tensor = (self.__model.forward(input_emotion, representation.diag())
                                             .to(dtype=self.__dtype, device=self.__device))

        attention_score: torch.tensor = torch.softmax(torch.sum(output, dim=1, dtype=self.__dtype),
                                                      dim=0, dtype=self.__dtype)

        new_representation: torch.tensor = torch.clamp(self.__weight(attention_score), -1, 1)

        return new_representation
