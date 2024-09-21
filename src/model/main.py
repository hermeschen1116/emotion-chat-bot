from fastapi import FastAPI

from model.libs.EmotionChatBot import EmotionChatBot

# get, post, patch, delete
app = FastAPI()


@app.on_event("startup")
def initialize():
	bot = EmotionChatBot()


@app.get("/")
def check_availability() -> bool:
	return True
