import torch


def generate_dummy_representation(target_emotion: int) -> torch.tensor:
    while True:
        dummy = torch.clamp(torch.rand(7, dtype=torch.float32), -1, 1)
        if torch.argmax(dummy) == target_emotion:
            return dummy


def get_sentiment_composition(sentiment_composition: list) -> torch.tensor:
    return torch.softmax(torch.tensor([result["score"] for result in sentiment_composition]), dim=-1)
