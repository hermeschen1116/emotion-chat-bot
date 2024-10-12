from typing import Dict

from torch import Tensor
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score


def calculate_evaluation_result(predictions: Tensor, truths: Tensor) -> Dict[str, Tensor]:
	accuracy: Tensor = multiclass_accuracy(predictions, truths, num_classes=7)
	f1_score: Tensor = multiclass_f1_score(predictions, truths, num_classes=7, average="weighted")

	return {"accuracy": accuracy, "f1_score": f1_score}
