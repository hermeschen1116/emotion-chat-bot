from typing import Any, Dict, List, Optional

from sympy import pprint

from .DataProcess import get_sentiment_composition


def get_top_emotion(input_text_emotion: list) -> str:
    label2score: dict = {
        emotion["label"]: emotion["score"] for emotion in input_text_emotion[0]
    }
    return max(label2score, key=label2score.get)


def create_candidates_buffer(chat_buffer: list) -> list:
    emotions: list = [
        "neutral",
        "anger",
        "disgust",
        "fear",
        "happiness",
        "sadness",
        "surprise",
    ]

    candidates_buffer: list = [
        chat_buffer
        + [
            {
                "role": "bot",
                "content": {"emotion": emotion, "dialog": ""},
            }
        ]
        for emotion in emotions
    ]

    return candidates_buffer


def get_possible_response_emotion_representation(
    candidates_buffer: list, emotion_predictor
) -> dict:
    possible_user_response_emotion: dict = {}
    for candidate in candidates_buffer:
        emotion: list = emotion_predictor(candidate[-1]["content"]["dialog"])
        possible_user_response_emotion[
            candidate[-1]["content"]["emotion"]
        ] = get_sentiment_composition(emotion)

    return possible_user_response_emotion


class ChatMessage:
    def __init__(self, system_prompt: Optional[str] = "", queue_size: int = 10) -> None:
        self.system_prompt: str = system_prompt
        self.queue_size: int = queue_size
        self.__message: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": {"dialog": system_prompt, "emotion": ""},
            }
        ]

    def append_message(
        self, dialog: Optional[str] = "", emotion: Optional[str] = ""
    ) -> None:
        if len(self.__message) == self.queue_size:
            self.__message.pop(1)

        self.__message.append(
            {
                "role": "user" if self.__message[-1]["role"] == "bot" else "bot",
                "content": {"dialog": dialog, "emotion": emotion},
            }
        )

    def get_message(self, index: int) -> Dict[str, Any]:
        return self.__message[index]

    def show_messages(self) -> None:
        pprint(self.__message)
