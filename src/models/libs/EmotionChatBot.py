import torch
from EmotionModel import EmotionModel
from ResponseGeneratorPipeline import ResponseGeneratorPipeline
from SimilarityAnalyser import SimilarityAnalyser
from transformers import BitsAndBytesConfig
from transformers.pipelines.text_classification import TextClassificationPipeline


class EmotionChatBot:
    def __init__(self, threshold: float) -> None:
        self.__response_generator = ResponseGeneratorPipeline()
        self.__sentiment_analyzer = TextClassificationPipeline(
            model="Shotaro30678/sentiment_analysis_for_emotion_chat_bot",
            framework="pt",
            task="sentiment-analysis",
            num_workers=16,
            device_map="auto",
            torch_dtype="auto",
            model_kwargs={
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
                ),
                "low_cpu_mem_usage": True,
            },
            trust_remote_code=True,
        )
        self.__emotion_predictor = TextClassificationPipeline()
        self.__emotion_model = EmotionModel.from_pretrained(
            "hermeschen1116/emotion_model_for_emotion_chat_bot"
        )
        self.__similarity_analyzer = SimilarityAnalyser(threshold=threshold)
