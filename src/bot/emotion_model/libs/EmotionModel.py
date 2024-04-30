import torch
import torch.nn.functional as nn
from lightning import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy

from .Attention import *


class EmotionModel(LightningModule):
    def __init__(self,
                 attention: str,
                 dropout: Optional[float] = 0.5,
                 scaler: Optional[float] = None,
                 bias: Optional[bool] = True,
                 dtype: Optional[Any] = torch.float32,
                 device: Optional[str] = "cpu"):
        super(EmotionModel, self).__init__()

        self.__dtype: Any = dtype
        self.__device: str = device

        match attention:
            case "dot_product":
                self.__attention = DotProductAttention()
            case "scaled_dot_product":
                self.__attention = ScaledDotProductAttention(scaler)
            case "additive":
                self.__attention = AdditiveAttention(dropout, dtype=dtype, device=device)
            case "dual_linear":
                self.__attention = DualLinearAttention(dropout, dtype=dtype, device=device)

        self.__weight = torch.nn.LazyLinear(7, bias=bias, device=device, dtype=dtype)

        self.__train_prediction: list = []
        self.__validation_prediction: list = []
        self.__test_prediction: list = []

    def forward(self, representation: torch.tensor, input_emotion: torch.tensor) -> torch.tensor:
        decomposed_representation: torch.tensor = representation.diag().to(dtype=self.__dtype, device=self.__device)

        output: torch.tensor = (self.__attention(input_emotion, decomposed_representation)
                                .to(dtype=self.__dtype, device=self.__device))

        attention_score: torch.tensor = torch.softmax(torch.sum(output, dim=1, dtype=self.__dtype),
                                                      dim=0, dtype=self.__dtype)

        difference: torch.tensor = torch.clamp(torch.sum(
            self.__weight((attention_score.diag()) ** 3), dim=1, dtype=self.__dtype), -1, 1)

        return representation + difference

    def representation_evolution(self, representation_src: list, emotion_compositions: list) -> list:
        representation: list = representation_src
        for composition in emotion_compositions:
            new_representation: torch.tensor = self.forward(
                torch.tensor(representation[-1], dtype=self.__dtype, device=self.__device),
                torch.tensor(composition, dtype=self.__dtype, device=self.__device))
            representation.append(new_representation)

        return representation

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer

    def on_train_epoch_start(self) -> None:
        self.__train_prediction.clear()

    def training_step(self, batch, batch_idx) -> dict:
        print(batch)
        data, label = batch

        output: list = self.representation_evolution(data[0], data[1])
        prediction: list = [torch.argmax(representation) for representation in output]
        self.__train_prediction.append({
            "prediction": torch.tensor(prediction),
            "truth": torch.tensor(label)
        })

        loss = nn.cross_entropy(torch.tensor(prediction), torch.tensor(label))

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return {"loss": loss.item()}

    def on_train_epoch_end(self) -> dict:
        all_prediction: torch.tensor = torch.cat([turn["prediction"] for turn in self.__train_prediction])
        all_truth: torch.tensor = torch.cat([turn["truth"] for turn in self.__train_prediction])

        f1_score = multiclass_f1_score(all_truth, all_prediction, num_classes=7, average="weighted")
        accuracy = multiclass_accuracy(all_truth, all_prediction, num_classes=7)

        self.log("train/f1_score", f1_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"f1_score": f1_score.item(), "accuracy": accuracy.item()}

    def on_validation_epoch_start(self) -> None:
        self.__validation_prediction.clear()

    def validation_step(self, batch, batch_idx) -> dict:
        data, label = batch

        output: list = self.representation_evolution(data[0], data[1])
        prediction: list = [torch.argmax(representation) for representation in output]
        self.__validation_prediction.append({
            "prediction": torch.tensor(prediction),
            "truth": torch.tensor(label)
        })

        loss = nn.cross_entropy(torch.tensor(prediction), torch.tensor(label))

        self.log("validation/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"validation_loss": loss.item()}

    def on_validation_epoch_end(self) -> dict:
        all_prediction: torch.tensor = torch.cat([turn["prediction"] for turn in self.__validation_prediction])
        all_truth: torch.tensor = torch.cat([turn["truth"] for turn in self.__validation_prediction])

        f1_score = multiclass_f1_score(all_truth, all_prediction, num_classes=7, average="weighted")
        accuracy = multiclass_accuracy(all_truth, all_prediction, num_classes=7)

        self.log("validation/f1_score", f1_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log("validation/accuracy", accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {"validation_f1_score": f1_score.item(), "validation_accuracy": accuracy.item()}
