from typing import Any, Dict, List, Optional, Self

from sympy import pprint

from .DataProcess import get_sentiment_composition


class ChatMessage:
    def __init__(self, system_prompt: str = "", queue_size: int = 10) -> None:
        self.system_prompt: str = system_prompt
        self.queue_size: int = queue_size
        self.__messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": {"dialog": system_prompt, "emotion": ""},
            }
        ]

    def append_message(
        self,
        dialog: Optional[str] = "",
        emotion: Optional[str] = "",
        inplace: bool = True,
    ) -> Self | None:
        messages: List[Dict[str, Any]] = self.__messages

        if len(messages) == self.queue_size:
            messages.pop(1)

        messages.append(
            {
                "role": "user" if self.__messages[-1]["role"] == "bot" else "bot",
                "content": {"dialog": dialog, "emotion": emotion},
            }
        )

        if inplace:
            self.__messages = messages
        else:
            return messages

    def get_message(self, index: int) -> Dict[str, Any]:
        return self.__messages[index]

    def show_messages(self) -> None:
        pprint(self.__messages)


def get_top_emotion(input_text_emotion: list) -> str:
    label2score: dict = {
        emotion["label"]: emotion["score"] for emotion in input_text_emotion[0]
    }
    return max(label2score, key=label2score.get)


def create_candidates_buffer(messages: ChatMessage) -> List[ChatMessage]:
    emotions: List[str] = [
        "neutral",
        "anger",
        "disgust",
        "fear",
        "happiness",
        "sadness",
        "surprise",
    ]

    candidates_buffer: list = [
        messages.append_message("", emotion, inplace=False) for emotion in emotions
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
