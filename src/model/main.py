from typing import Dict

from fastapi import FastAPI
from libs.EmotionChatBot import EmotionChatBot

# get, post, patch, delete
service = FastAPI(title="Emotion-Chat-Bot")


@service.on_event("startup")
def initialize_service():
	bot = EmotionChatBot()


@service.get("/")
def check_service_availability() -> Dict[str, bool]:
	return {"servic_availability": True}
