from typing import Any, Dict, List, Optional, Union

import torch
from torch import Tensor
from transformers import (
	AutoModelForSequenceClassification,
	AutoTokenizer,
	BitsAndBytesConfig,
	TextClassificationPipeline, GenerationConfig,
)
from unsloth import FastLanguageModel

from src.models.libs.EmotionTransition import (
	EmotionModel,
	SimilarityAnalyzer,
	generate_representation,
	get_emotion_composition,
)
from src.models.libs.ResponseGenerationPipeline import ResponseGeneratorPipeline

roles: List[str] = ["user", "bot"]
emotions: List[str] = [
	"neutral",
	"anger",
	"disgust",
	"fear",
	"happiness",
	"sadness",
	"surprise",
]

default_generation_config = GenerationConfig(
	max_new_tokens=20,
	min_new_tokens=5,
	repetition_penalty=1.5,
	pad_token_id=32000,
	eos_token_id=2,
)


class EmotionChatBot:
	def __init__(
		self,
		sentiment_analyzer_model_name: str = "Shotaro30678/sentiment_analysis_for_emotion_chat_bot",
		emotion_predictor_model_name: str = "Shotaro30678/emotion_predictor_for_emotion_chat_bot",
		emotion_model_model_name: str = "hermeschen1116/emotion_model_for_emotion_chat_bot",
		response_generator_model_name: str = "hermeschen1116/response_generator_for_emotion_chat_bot",
		similarity_threshold: Optional[float] = 0.7,
		emotion_tendency: Optional[Union[Dict[str, float], int]] = None,
		system_prompt: Optional[str] = "",
		max_num_turns: Optional[int] = 5,
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

		self.system_prompt: Dict[str, Any] = {"role": "system", "content": {"emotion": "", "dialog": system_prompt}}
		self.max_num_messages: int = max_num_turns * 2
		self.messages: List[Dict[str, Any]] = []
		self.emotion_representation: Tensor = generate_representation(None)

	@staticmethod
	def __form_message(role: str, emotion: Optional[str] = "", dialog: Optional[str] = "") -> Dict[str, Any]:
		return {"role": role, "content": {"emotion": emotion, "dialog": dialog}}

	def __append_message(self, emotion: str, dialog: Optional[str] = "") -> None:
		next_role: str = roles[len(self.messages) % 2]

		self.messages.append(self.__form_message(next_role, emotion, dialog))

		if len(self.messages) > self.max_num_messages:
			self.messages.pop(0)

	def __get_messages(self) -> List[Dict[str, Any]]:
		return [self.system_prompt] + self.messages

	def __process_user_emotion(self, message: str) -> (Tensor, str):
		sentiment_analysis_result: list = self.sentiment_analyzer(message, return_all_scores=True)

		user_emotion_composition: Tensor = get_emotion_composition(sentiment_analysis_result[0])
		user_emotion: str = emotions[user_emotion_composition.argmax()]

		return user_emotion_composition, user_emotion

	def __create_candidate_chats(self) -> List[List[Dict[str, Any]]]:
		messages: list = self.__get_messages()
		candidates_chats: list = [
			messages + [{"role": "bot", "content": {"emotion": emotion, "dialog": ""}}] for emotion in emotions
		]

		return candidates_chats

	def __generate_candidate_responses(self, generation_config) -> Dict[str, str]:
		candidates_chats: list = self.__create_candidate_chats()
		candidates_chats = [
			chat[0] for chat in self.response_generator(candidates_chats, generation_config=generation_config)
		]

		candidates_responses: list = [chat[-1]["content"]["dialog"].strip() for chat in candidates_chats]

		return dict(zip(emotions, candidates_responses))

	@staticmethod
	def __validate_response(response: str) -> bool:
		return (len(response) != 0) and response.endswith((".", "!", "?"))

	def __predict_user_response_emotion(self, candidate_responses: Dict[str, str]) -> Dict[str, Tensor]:
		return {
			k: get_emotion_composition(self.emotion_predictor(v, return_all_scores=True)[0])
			for k, v in candidate_responses.items()
		}

	def __call__(self, user_message: str, generation_config: Optional[GenerationConfig] = default_generation_config) -> Dict[str, str]:
		user_emotion_composition, user_emotion = self.__process_user_emotion(user_message)

		self.__append_message(user_emotion, user_message)

		self.bot_emotion_representation = self.emotion_model.forward(
			user_emotion_composition, self.emotion_representation
		)

		while True:
			candidates_responses = self.__generate_candidate_responses(generation_config)
			candidates_responses = dict(filter(lambda item: self.__validate_response(item[1]), candidates_responses.items()))
			if len(candidates_responses.keys()) != 0:
				break

		user_future_emotion_compositions: dict = self.__predict_user_response_emotion(candidates_responses)

		future_emotion_representations: dict = {
			k: self.emotion_model.forward(v, self.emotion_representation)
			for k, v in user_future_emotion_compositions.items()
		}

		emotion_representation_similarity_scores: Tensor = self.similarity_analyzer(
			torch.stack(list(future_emotion_representations.values()))
		)
		best_response_emotion_index: int = emotion_representation_similarity_scores.argmax().item()
		best_response_emotion: str = list(future_emotion_representations.keys())[best_response_emotion_index]

		self.__append_message(best_response_emotion)
		self.messages = self.response_generator(self.__get_messages(), generation_config=generation_config)[0][1:]
		response: str = self.messages[-1]["content"]["dialog"]
		if not self.__validate_response(response):
			self.messages[-1]["content"]["dialog"] = list(candidates_responses.values())[best_response_emotion_index]

		return {"emotion": best_response_emotion, "response": self.messages[-1]["content"]["dialog"]}
