from datasets import load_dataset

from emotion_chat_bot.utils.Helper import login_to_service

login_to_service()

dataset = load_dataset("daily_dialog", num_proc=16, save_infos=True, trust_remote_code=True).remove_columns("act")

dataset = dataset.rename_column("emotion", "emotion_id")
emotion_labels: list = dataset["train"].features["emotion_id"].feature.names
emotion_labels[0] = "neutral"
dataset = dataset.map(
	lambda samples: {"emotion": [[emotion_labels[emotion_id] for emotion_id in sample] for sample in samples]},
	input_columns="emotion_id",
	remove_columns="emotion_id",
	batched=True,
	num_proc=16,
)

dataset = dataset.map(
	lambda samples: {
		"emotion": [sample[:-1] if len(sample) % 2 == 1 else sample for sample in samples["emotion"]],
		"dialog": [sample[:-1] if len(sample) % 2 == 1 else sample for sample in samples["dialog"]],
	},
	batched=True,
	num_proc=16,
)

dataset = dataset.filter(lambda sample: (len(sample["emotion"]) != 0) and (len(sample["dialog"]) != 0), num_proc=16)

dataset = dataset.map(
	lambda samples: {
		"prompt": [
			[
				{
					"role": "user" if i % 2 == 0 else "assistant",
					"content": {"emotion": emotion, "dialog": dialog.strip()},
				}
				for i, (emotion, dialog) in enumerate(zip(sample[0], sample[1]))
			]
			for sample in zip(samples["emotion"], samples["dialog"])
		]
	},
	remove_columns=["emotion", "dialog"],
	batched=True,
	num_proc=16,
)

dataset.push_to_hub("daily_dialog_for_RG", num_shards={"train": 16, "validation": 16, "test": 16})
