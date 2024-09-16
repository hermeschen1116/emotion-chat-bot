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


def flatten_data_and_abandon_data_with_neutral(
    source_dataset: Dataset, keep_ratio: float
) -> Dataset:
    flattened_dataset = Dataset.from_list(
        [
            row
            for sample in source_dataset["rows"]
            for row in sample
        ]
    )

    dataset_without_neutral = flattened_dataset.filter(lambda sample: sample != 0, input_columns=["label"], num_proc=16)
    dataset_with_only_neutral = flattened_dataset.filter(lambda sample: sample == 0, input_columns=["label"], num_proc=16)
    num_row_with_neutral_to_take: int = int(len(dataset_with_only_neutral) * keep_ratio)

    return concatenate_datasets(
        [
            dataset_without_neutral,
            dataset_with_only_neutral.shuffle().take(num_row_with_neutral_to_take),
        ]
    )
