from typing import Any, Dict, List


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