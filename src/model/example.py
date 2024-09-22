from libs.EmotionChatBot import EmotionChatBot

# 情緒傾向（會依此生成 Ideal Emotion Representation）
# emotion_tendency: dict = {
# 	"neutral": 0.2,
# 	"anger": 0.4,
# 	"disgust": 0.5,
# 	"fear": 0.1,
# 	"happiness": 0.8,
# 	"sadness": 0.2,
# 	"surprise": 0.2,
# }
# 預設是隨機產生
emotion_tendency = None

bot = EmotionChatBot(emotion_tendency=emotion_tendency)

while True:
	user_message: str = input("User: ").strip()

	result = bot(user_message)

	print(f"Bot: {result['response']}")
