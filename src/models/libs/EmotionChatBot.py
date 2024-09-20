from typing import Any, Dict, List, Literal, Optional, Union

import torch
from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
	BitsAndBytesConfig,
	TextClassificationPipeline,
)
from unsloth import FastLanguageModel

from src.models.libs.EmotionTransition import EmotionModel, SimilarityAnalyzer, generate_representation
from src.models.libs.ResponseGenerationPipeline import ResponseGeneratorPipeline

role: List[str] = ["user", "bot"]
emotions: List[str] = [
	"neutral",
	"anger",
	"disgust",
	"fear",
	"happiness",
	"sadness",
	"surprise",
]


class Chat:
	def __init__(self, system_prompt: Optional[str] = "", max_num_turns: Optional[int] = 5) -> None:
		self.__messages: List[Dict[str, Any]] = []
		self.__max_num_messages: int = max_num_turns * 2
		self.__system_prompt: str = system_prompt

	@property
	def system_prompt(self) -> str:
		return self.__system_prompt

	@system_prompt.setter
	def system_prompt(self, system_prompt: str) -> None:
		self.__system_prompt = system_prompt

	def get_system_message(self) -> Dict[str, Any]:
		return {"role": "system", "content": {"emotion": "", "dialog": self.system_prompt}}

	@property
	def messages(self) -> List[Dict[str, Any]]:
		return [self.get_system_message] + self.__messages

	@property.setter
	def messages(self, messages: List[Dict[str, Any]]) -> None:
		if len(messages) > self.__max_num_messages:
			raise ValueError(
				f"Cannot assign messages over maximum number of dialogs {self.__max_num_messages}, you input length is {len(messages)}.\n"
			)
		self.__messages = messages

	def append_message(
		self, emotion: str, dialog: Optional[str] = "", inplace: Optional[bool] = True
	) -> Optional[Dict[str, Any]]:
		if emotion not in emotions:
			raise ValueError(f"Input emotion {emotion} is not a valid emotion.")

		next_role: Literal["user", "bot"] = role[(len(self.__messages) - 1) % 2]

		messages: Dict[str, Any] = self.__messages

		messages.append({"role": next_role, "content": {"emotion": emotion, "dialog": dialog}})

		if len(messages) > self.__max_num_turns:
			while len(messages) > self.__max_num_turns:
				messages.pop(0)

		if inplace:
			self.__messages = messages

		return [self.get_system_message()] + messages


def create_candidate_chats(
	chat: Chat,
) -> List[List[Dict[str, Any]]]:
	candidates_chats: list = [
		chat.messages + [{"role": "bot", "content": {"emotion": emotion, "dialog": ""}}] for emotion in emotions
	]

	return candidates_chats


class EmotionChatBot:
	def __init__(
		self,
		sentiment_analyzer_model_name: str = "Shotaro30678/sentiment_analysis_for_emotion_chat_bot",
		emotion_predictor_model_name: str = "Shotaro30678/emotion_predictor_for_emotion_chat_bot",
		emotion_model_model_name: str = "hermeschen1116/emotion_model_for_emotion_chat_bot",
		response_generator_model_name: str = "hermeschen1116/response_generator_for_emotion_chat_bot",
		similarity_threshold: Optional[float] = 0.7,
		emotion_tendency: Optional[Union[Dict[str, float], int]] = None,
	) -> None:
		model, tokenizer = FastLanguageModel.from_pretrained(
			model_name=response_generator_model_name,
			attn_implementation="flash_attention_2",
			pretraining_tp=1,
			load_in_4bit=True,
			device_map="auto",
			low_cpu_mem_usage=True,
			trust_remote_code=True,
		)
		tokenizer.padding_side = "left"
		tokenizer.clean_up_tokenization_spaces = True
		FastLanguageModel.for_inference(model)

		self.response_generator = ResponseGeneratorPipeline(
			model,
			tokenizer,
			framework="pt",
			task="conversation-generation",
			num_workers=16,
			torch_dtype="auto",
			add_special_tokens=True,
			truncation=False,
			padding=True,
		)

		sentiment_analysis_model = AutoModelForSequenceClassification.from_pretrained(
			sentiment_analyzer_model_name,
			quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
			device_map="auto",
			low_cpu_mem_usage=True,
		)

		sentiment_analysis_tokenizer = AutoTokenizer.from_pretrained(
			sentiment_analyzer_model_name,
			trust_remote_code=True,
		)

		self.sentiment_analyzer = TextClassificationPipeline(
			model=sentiment_analysis_model,
			tokenizer=sentiment_analysis_tokenizer,
			framework="pt",
			task="sentiment-analysis",
			num_workers=12,
			torch_dtype="auto",
		)

		emotion_predictor_model = AutoModelForSequenceClassification.from_pretrained(
			emotion_predictor_model_name,
			quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16),
			device_map="auto",
			low_cpu_mem_usage=True,
		)

		emotion_predictor_tokenizer = AutoTokenizer.from_pretrained(
			emotion_predictor_model_name,
			trust_remote_code=True,
		)

		self.emotion_predictor = TextClassificationPipeline(
			model=emotion_predictor_model,
			tokenizer=emotion_predictor_tokenizer,
			framework="pt",
			task="sentiment-analysis",
			num_workers=12,
			torch_dtype="auto",
		)

		self.emotion_model = EmotionModel.from_pretrained(emotion_model_model_name)

		self.similarity_analyzer = SimilarityAnalyzer(
			generate_representation(emotion_tendency), threshold=similarity_threshold
		)
