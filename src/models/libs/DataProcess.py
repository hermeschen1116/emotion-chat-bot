from typing import Any

import torch
from datasets import Dataset, concatenate_datasets
from torch import Tensor


def generate_dummy_representation(target_emotion: int) -> Tensor:
    while True:
        dummy = torch.clamp(torch.rand(7, dtype=torch.float32), -1, 1)
        if torch.argmax(dummy) == target_emotion:
            return dummy


def get_sentiment_composition(analysis_result: list) -> Tensor:
    sentiment_possibility: dict = {}
    for emotion_score in analysis_result[0]:
        values: list = list(emotion_score.values())
        sentiment_possibility[values[0]] = values[1]

    emotions: list = [
        "neutral",
        "anger",
        "disgust",
        "fear",
        "happiness",
        "sadness",
        "surprise",
    ]
    sentiment_composition: list = [
        sentiment_possibility[emotion] for emotion in emotions
    ]

    return torch.softmax(
        torch.tensor(sentiment_composition), dim=-1, dtype=torch.float32
    )


def flatten_dataset(dataset: Dataset, target_column_name: str = "rows") -> Dataset:
    return Dataset.from_list(
        [row for sample in dataset[target_column_name] for row in sample]
    )


def throw_out_partial_row_with_a_label(
    dataset: Dataset, keep_ratio: float, filter_value: Any, label_name: str = "label"
) -> Dataset:
    dataset_without_label = dataset.filter(
        lambda sample: sample != filter_value, input_columns=[label_name], num_proc=16
    )
    dataset_with_label = dataset.filter(
        lambda sample: sample == filter_value, input_columns=[label_name], num_proc=16
    )
    num_row_with_label_to_keep: int = int(len(dataset_with_label) * keep_ratio)

    return concatenate_datasets(
        [
            dataset_without_label,
            dataset_without_label.shuffle().take(num_row_with_label_to_keep),
        ]
    ).shuffle()
