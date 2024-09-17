import json
from datetime import datetime
from typing import Any, Dict, List
from uuid import UUID


def get_top_emotion(input_text_emotion: list) -> str:
	label2score: dict = {emotion["label"]: emotion["score"] for emotion in input_text_emotion[0]}
	return max(label2score, key=label2score.get)


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


def write_log_file(run_uuid: UUID, content: Any) -> None:
	with open(f"logs/{run_uuid}.log", "w", encoding="utf-8") as log_file:
		log_file.write(json.dumps({"timestamp": str(datetime.now()), "content": content}))
