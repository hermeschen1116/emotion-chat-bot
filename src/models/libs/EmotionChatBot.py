from typing import Any, Dict, List, Optional, Union

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


def create_candidates_buffer(
	messages: List[Dict[str, Any]],
) -> List[List[Dict[str, Any]]]:
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
		messages + [{"role": "bot", "content": {"emotion": emotion, "dialog": ""}}] for emotion in emotions
	]

	return candidates_buffer


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
