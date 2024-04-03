import argparse
import os

import huggingface_hub
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
from tqdm.auto import tqdm
from transformers import (AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig,
                          GenerationConfig, pipeline)

import wandb

# prevent env load failed
load_dotenv(encoding="utf-8")

huggingface_hub.login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=True)

wandb.login(key=os.environ.get("WANDB_API_KEY", ""), relogin=True)

parser = argparse.ArgumentParser(prog="Evaluation", description="Evaluation Script For Response Generator")
parser.add_argument("--base_model", required=True, type=str)
parser.add_argument("--fine_tuned_model", required=False, type=str, default="")
parser.add_argument("--prompt_type", required=False, type=str, default="")
arguments = parser.parse_args()
arguments.fine_tuned_model = arguments.base_model if arguments.fine_tuned_model == "" else arguments.fine_tuned_model

# Load and Process Dataset
dataset = load_dataset("daily_dialog", split="test", num_proc=8, trust_remote_code=True)

dataset = dataset.remove_columns("act")

dataset = dataset.rename_column("emotion", "emotion_id")
emotion_labels: list = dataset.features["emotion_id"].feature.names
emotion_labels[0] = "neutral"
dataset = dataset.map(
    lambda samples: {"emotion": [[emotion_labels[emotion_id] for emotion_id in sample] for sample in samples]},
    input_columns="emotion_id", remove_columns="emotion_id", batched=True, num_proc=8)

dataset = dataset.map(lambda samples: {"emotion_user": [sample[:-1] for sample in samples]}, input_columns="emotion",
                      batched=True, num_proc=8)

dataset = dataset.map(lambda samples: {"emotion_bot": [sample[1:] for sample in samples]}, input_columns="emotion",
                      batched=True, num_proc=8)

dataset = dataset.remove_columns("emotion")

dataset = dataset.map(
    lambda samples: {"dialog_user": [[dialog.strip() for dialog in sample][:-1] for sample in samples]},
    input_columns="dialog", batched=True, num_proc=8)

dataset = dataset.map(lambda samples: {"dialog_bot": [[dialog.strip() for dialog in sample][1:] for sample in samples]},
                      input_columns="dialog", batched=True, num_proc=8)

dataset = dataset.remove_columns("dialog")

# Build Up Prompt
enter: str = "\n"
dataset = dataset.map(lambda samples: {"prompts": [
    [f"""<s>[INST]user({emotion_user}): {dialog_user}{enter}bot({emotion_bot}): [/INST]""" for
     emotion_user, emotion_bot, dialog_user in zip(sample[0], sample[1], sample[2])] for sample in
    zip(samples["emotion_user"], samples["emotion_bot"], samples["dialog_user"])]}, remove_columns="emotion_user",
                      batched=True, num_proc=8)

test_data: list = []
for sample in tqdm(dataset.to_list(), colour="green"):
    for i in range(len(sample['emotion_bot'])):
        test_data.append({"prompt": sample["prompts"][i], "emotion_bot": sample["emotion_bot"][i],
                          "dialog_user": sample["dialog_user"][i], "dialog_bot": sample["dialog_bot"][i]})

# Load Model
device_map: str = "auto" if torch.cuda.is_available() else "cpu"

quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

model = AutoModelForCausalLM.from_pretrained(arguments.fine_tuned_model,
                                             quantization_config=quantization_config if torch.cuda.is_available() else None,
                                             device_map=device_map, low_cpu_mem_usage=True,
                                             attn_implementation="flash_attention_2")

model = torch.compile(model)

# Configuration
generation_config = GenerationConfig(max_new_tokens=20, min_new_tokens=5, repetition_penalty=1.5)

# Log
wandb.init(project="emotion-chat-bot-ncu", group="Response Generator", job_type="evaluation",
           config={"base_model": arguments.base_model, "fine_tuned_model": arguments.fine_tuned_model,
                   "quantization_configuration": quantization_config.to_dict(),
                   "generation_configuration": generation_config.to_dict(),
                   "experiment_detail": "evaluate some possible base model"})

# Generate Response
tokenizer = AutoTokenizer.from_pretrained(arguments.base_model, trust_remote_code=True)
generation_config.pad_token_id = tokenizer.eos_token_id

device: str = "cuda" if torch.cuda.is_available() else "cpu"
for sample in tqdm(test_data, colour="green"):
    tokenized_prompt = tokenizer(sample["prompt"], return_tensors="pt").to(device)
    response_ids = model.generate(**tokenized_prompt, generation_config=generation_config)
    response = tokenizer.decode(response_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    sample["response_bot"] = response.replace(response.split("[/INST]")[0], "").removeprefix("[/INST]").strip()

result = dataset.from_list(test_data)

# Sentiment Analysis
sentiment_analysis_model = AutoModelForSequenceClassification.from_pretrained(
    "../sentiment _analysis/emotion_text_classifier_on_dd_v1",
    quantization_config=quantization_config if torch.cuda.is_available() else None, device_map=device_map,
    low_cpu_mem_usage=True)
# sentiment_analysis_model = torch.compile(sentiment_analysis_model)


sentiment_analysis_tokenizer = AutoTokenizer.from_pretrained("michellejieli/emotion_text_classifier",
                                                             trust_remote_code=True)

analyser = pipeline("sentiment-analysis", model=sentiment_analysis_model, tokenizer=sentiment_analysis_tokenizer,
                    device_map=device_map, trust_remote_code=True)

# to prevent "The model 'OptimizedModule' is not supported for sentiment-analysis." problem
sentiment_analysis_model = torch.compile(sentiment_analysis_model)

result = result.map(lambda samples: {"response_sentiment_bot": analyser(samples)}, input_columns="response_bot",
                    batched=True)

# Metrics
emotion_label_to_id: dict = {label: index for index, label in enumerate(emotion_labels)}

result = result.map(lambda samples: {"emotion_bot_id": emotion_label_to_id[samples]}, input_columns="emotion_bot",
                    num_proc=8)

result = result.map(lambda samples: {"response_sentiment_bot": samples["label"]
                    if samples["label"] != "joy" else "happiness"},
                    input_columns="response_sentiment_bot", num_proc=8)

result = result.map(lambda samples: {"response_sentiment_bot_id": emotion_label_to_id[samples["label"]]
                    if samples["label"] in emotion_label_to_id.keys() else 4},
                    input_columns="response_sentiment_bot", num_proc=8)

# Result
sentiment_true: torch.tensor = torch.tensor(
    [sample["emotion_bot_id"] for sample in result.select_columns("emotion_bot_id").to_list()])
sentiment_pred: torch.tensor = torch.tensor(
    [sample["response_sentiment_bot_id"] for sample in result.select_columns("response_sentiment_bot_id").to_list()])

result = result.remove_columns(["emotion_bot_id", "response_sentiment_bot_id"])

f1 = multiclass_f1_score(sentiment_true, sentiment_pred, num_classes=len(emotion_labels), average="micro")

accuracy = multiclass_accuracy(sentiment_true, sentiment_pred, num_classes=len(emotion_labels))

wandb.log({"F1-score": f1, "Accuracy": accuracy})

wandb.log({"generated_test_result": wandb.Table(dataframe=result.to_pandas())})

wandb.finish()
