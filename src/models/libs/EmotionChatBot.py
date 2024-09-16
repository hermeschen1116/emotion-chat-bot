from transformers.pipelines.text_classification import TextClassificationPipeline

from models.libs.EmotionModel import EmotionModel
from .ResponseGeneratorPipeline import ResponseGeneratorPipeline

class EmotionChatBot:
	def __init__(self) -> None:
		self.__response_generator = ResponseGeneratorPipeline()
		self.__sentiment_analyzer = TextClassificationPipeline()
		self.__emotion_predictor = TextClassificationPipeline()
		self.__emotion_model = EmotionModel.from_pretrained()
