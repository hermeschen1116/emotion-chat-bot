from typing import Any

import torch
from datasets import Dataset, concatenate_datasets
from torch import Tensor


def flatten_dataset(dataset: Dataset, target_column_name: str = "rows") -> Dataset:
	return Dataset.from_list([row for sample in dataset[target_column_name] for row in sample])


def throw_out_partial_row_with_a_label(
	dataset: Dataset, keep_ratio: float, filter_value: Any, label_name: str = "label"
) -> Dataset:
	dataset_without_label = dataset.filter(
		lambda sample: sample != filter_value, input_columns=[label_name], num_proc=16
	)
	dataset_with_label = dataset.filter(lambda sample: sample == filter_value, input_columns=[label_name], num_proc=16)
	num_row_with_label_to_keep: int = int(len(dataset_with_label) * keep_ratio)

	return concatenate_datasets(
		[
			dataset_without_label,
			dataset_with_label.shuffle().take(num_row_with_label_to_keep),
		]
	).shuffle()
