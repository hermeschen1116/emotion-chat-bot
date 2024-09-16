import torch
from EmotionModel import EmotionModel
from ResponseGeneratorPipeline import ResponseGeneratorPipeline
from SimilarityAnalyser import SimilarityAnalyser
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from unsloth import FastLanguageModel


class EmotionChatBot:
    def __init__(self, threshold: float) -> None:
	   	model, tokenizer = FastLanguageModel.from_pretrained(
	        model_name="Shotaro30678/response_generator_DPO",  # YOUR MODEL YOU USED FOR TRAINING
	        attn_implementation="flash_attention_2",
		    pretraining_tp=1,
		    load_in_4bit=True,
		    device_map="auto",
		    low_cpu_mem_usage=True,
		    trust_remote_code=True,
	    )

		FastLanguageModel.for_inference(model)

		self.__response_generator = ResponseGeneratorPipeline(
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
            "Shotaro30678/sentiment_analysis_for_emotion_chat_bot",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            ),
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        sentiment_analysis_tokenizer = AutoTokenizer.from_pretrained(
            "Shotaro30678/sentiment_analysis_for_emotion_chat_bot",
            trust_remote_code=True,
        )

        sentiment_analysis_model = torch.compile(sentiment_analysis_model)
        self.__sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=sentiment_analysis_model,
            tokenizer=sentiment_analysis_tokenizer,
            top_k=7,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        emotion_predictor_model = AutoModelForSequenceClassification.from_pretrained(
            "Shotaro30678/sentiment_analysis_for_emotion_chat_bot",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
            ),
            device_map="auto",
            low_cpu_mem_usage=True,
        )

        emotion_predictor_tokenizer = AutoTokenizer.from_pretrained(
            "Shotaro30678/sentiment_analysis_for_emotion_chat_bot",
            trust_remote_code=True,
        )

        emotion_predictor_model = torch.compile(sentiment_analysis_model)
        self.__emotion_predictor = pipeline(
            "sentiment-analysis",
            model=emotion_predictor_model,
            tokenizer=emotion_predictor_tokenizer,
            top_k=7,
            torch_dtype=torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.__emotion_model = EmotionModel.from_pretrained(
            "hermeschen1116/emotion_model_for_emotion_chat_bot"
        )
        self.__similarity_analyzer = SimilarityAnalyser(threshold=threshold)
