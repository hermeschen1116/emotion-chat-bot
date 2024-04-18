import argparse
import os
import shutil

import huggingface_hub
import wandb
from datasets import load_dataset
from dotenv import load_dotenv

parser = argparse.ArgumentParser(prog="Evaluation", description="Evaluation Script For Response Generator")
parser.add_argument("--dataset_path", required=True, type=str, default="./dataset")
parser.add_argument("--note", required=False, type=str, default="")
parser.add_argument("--wandb_mode",
                    required=False,
                    type=str,
                    choices=["online", "offline", "disabled"],
                    default="online")
arguments = parser.parse_args()

load_dotenv(encoding="utf-8")
huggingface_hub.login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=True)
wandb.login(key=os.environ.get("WANDB_API_KEY", ""), relogin=True)

run = wandb.init(
    job_type="dataset",
    project="emotion-chat-bot-ncu",
    group="Response Generator",
    notes=arguments.note,
    mode=arguments.wandb_mode,
    resume="auto"
)

dataset = load_dataset("daily_dialog",
                       num_proc=16,
                       save_infos=True,
                       trust_remote_code=True).remove_columns("act")

dataset = dataset.rename_column("emotion", "emotion_id")
emotion_labels: list = dataset["train"].features["emotion_id"].feature.names
emotion_labels[0] = "neutral"
dataset = dataset.map(lambda samples: {
    "emotion": [[emotion_labels[emotion_id] for emotion_id in sample] for sample in samples]
}, input_columns="emotion_id", remove_columns="emotion_id", batched=True, num_proc=16)

dataset = dataset.map(lambda samples: {
    "dialog": [[dialog.strip() for dialog in sample] for sample in samples]
}, input_columns="dialog", batched=True, num_proc=16)

dataset = dataset.map(lambda samples: {
    "emotion": [sample[:-1] if len(sample) % 2 == 1 else sample for sample in samples["emotion"]],
    "dialog": [sample[:-1] if len(sample) % 2 == 1 else sample for sample in samples["dialog"]]
}, batched=True, num_proc=16)

dataset["test"] = dataset["test"].map(lambda samples: {
    "emotion_history": [sample[:-1] for sample in samples],
    "emotion_bot": [sample[-1] for sample in samples]
}, input_columns="emotion", batched=True, num_proc=16)

dataset["test"] = dataset["test"].map(lambda samples: {
    "dialog_history": [sample[:-1] for sample in samples],
    "dialog_bot": [sample[-1] for sample in samples]
}, input_columns="dialog", batched=True, num_proc=16)

dataset = dataset.map(lambda samples: {
    "prompt": [[{
        "role": "user" if i % 2 == 0 else "assistant",
        "content": {"emotion": emotion, "dialog": dialog}}
        for i, (emotion, dialog) in enumerate(zip(sample[0], sample[1]))]
        for sample in zip(samples["emotion"], samples["dialog"])]
}, remove_columns=["emotion", "dialog"], batched=True, num_proc=16)

dataset["test"] = dataset["test"].map(lambda samples: {
    "prompt": [[{
        "role": "user" if i % 2 == 0 else "assistant",
        "content": {"emotion": emotion, "dialog": dialog}}
        for i, (emotion, dialog) in enumerate(zip(sample[0], sample[1]))]
        for sample in zip(samples["emotion_history"], samples["dialog_history"])]
}, batched=True, num_proc=16)

dataset["test"] = dataset["test"].map(lambda sample: {
    "history": [
        "\n".join([f"""{'user' if i % 2 == 0 else 'bot'}({v[0]}): {v[1]}"""
                   for i, v in enumerate(zip(sample["emotion_history"], sample["dialog_history"]))])
    ]}, remove_columns=["emotion_history", "dialog_history"], num_proc=16)

test_dataset_artifact = wandb.Artifact(
    "daily_dialog_for_RG_test",
    type="dataset",
    description="modified version of daily dialog dataset from huggingface for response generator module",
    metadata=dict(dataset),
    incremental=True
)

dataset["test"].save_to_disk(f"{arguments.dataset_path}_test", num_proc=16)
test_dataset_artifact.add_dir(f"{arguments.dataset_path}_test")

run.log_artifact(test_dataset_artifact)

dataset_artifact = wandb.Artifact(
    "daily_dialog_for_RG_train",
    type="dataset",
    description="modified version of daily dialog dataset from huggingface for response generator module",
    metadata=dict(dataset),
    incremental=True
)

dataset["train"].save_to_disk(f"{arguments.dataset_path}_train", num_proc=16)
dataset_artifact.add_dir(f"{arguments.dataset_path}_train")

run.log_artifact(dataset_artifact)

wandb.finish()

if os.path.exists(f"{arguments.dataset_path}_train"):
    shutil.rmtree(f"{arguments.dataset_path}_train", ignore_errors=True)
if os.path.exists(f"{arguments.dataset_path}_test"):
    shutil.rmtree(f"{arguments.dataset_path}_test", ignore_errors=True)
