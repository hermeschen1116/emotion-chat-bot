import argparse
import os

import huggingface_hub
import torch
from datasets import load_dataset
from dotenv import load_dotenv
from torcheval.metrics.functional import (multiclass_accuracy,
                                          multiclass_f1_score)
from tqdm.auto import tqdm
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          GenerationConfig,
                          pipeline)

import wandb
# from Pipeline import ResponseGeneratorPipeline

# prevent env load failed
load_dotenv(encoding="utf-8")
huggingface_hub.login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=True)
wandb.login(key=os.environ.get("WANDB_API_KEY", ""), relogin=True)

# commandline inputs
parser = argparse.ArgumentParser(prog="Evaluation", description="Evaluation Script For Response Generator")
parser.add_argument("--base_model", required=True, type=str)
parser.add_argument("--fine_tuned_model", required=False, type=str, default="")
parser.add_argument("--prompt_type", required=False, type=str, default="")
parser.add_argument("--sentiment_analysis_tokenizer",
                    required=False,
                    type=str,
                    default="michellejieli/emotion_text_classifier")
parser.add_argument("--sentiment_analysis_model",
                    required=False,
                    type=str,
                    default="michellejieli/emotion_text_classifier")
parser.add_argument("--experiment_detail", required=True, type=str, default="")

arguments = parser.parse_args()
arguments.fine_tuned_model = arguments.base_model if arguments.fine_tuned_model == "" else arguments.fine_tuned_model

print(f"base model: {arguments.base_model}, fine-tuned model: {arguments.fine_tuned_model}")
# Initialize Wandb
wandb.init(project="emotion-chat-bot-ncu",
           group="Response Generator",
           job_type="evaluation",
           config={"base_model": arguments.base_model,
                   "fine_tuned_model": arguments.fine_tuned_model,
                   "experiment_detail": arguments.experiment_detail})

# Load and Process Dataset
dataset = load_dataset("daily_dialog", split="test", num_proc=16, trust_remote_code=True)
dataset = dataset.remove_columns("act")
dataset = dataset.rename_column("emotion", "emotion_id")
emotion_labels: list = dataset.features["emotion_id"].feature.names
emotion_labels[0] = "neutral"
dataset = dataset.map(lambda samples: {
    "emotion": [[emotion_labels[emotion_id] for emotion_id in sample] for sample in samples]
}, input_columns="emotion_id", remove_columns="emotion_id", batched=True, num_proc=16)
dataset = dataset.map(lambda samples: {
    "emotion_history": [sample[:-1] for sample in samples],
    "emotion_bot": [sample[-1] for sample in samples]
}, input_columns="emotion", remove_columns="emotion", batched=True, num_proc=16)
dataset = dataset.map(lambda samples: {
    "dialog": [[dialog.strip() for dialog in sample] for sample in samples]
}, input_columns="dialog", batched=True, num_proc=16)
dataset = dataset.map(lambda samples: {
    "dialog": [sample if len(sample) % 2 == 0 else sample[:-1] for sample in samples]
}, input_columns="dialog", batched=True, num_proc=16)
dataset = dataset.map(lambda samples: {
    "dialog_history": [sample[:-1] for sample in samples],
    "dialog_bot": [sample[-1] for sample in samples]
}, input_columns="dialog", remove_columns="dialog", batched=True, num_proc=16)
bot: str = "model" if 'gemma' in arguments.base_model.lower() else "assistant"
test_data = dataset.map(lambda samples: {
    "prompt": [[{"role": "user" if i % 2 == 0 else "assistant", "content": dialog}
                for i, dialog in enumerate(sample)] for sample in samples]
}, input_columns="dialog_history", batched=True, num_proc=16)


device_map: str = "auto" if torch.cuda.is_available() else "cpu"
# Load Model
quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                         bnb_4bit_compute_dtype=torch.float16)
quantization_config = quantization_config if torch.cuda.is_available() else None
wandb.config["quantization_configuration"] = quantization_config.to_dict() if quantization_config is not None else {}

model = AutoModelForCausalLM.from_pretrained(arguments.fine_tuned_model,
                                             quantization_config=quantization_config,
                                             attn_implementation="flash_attention_2",
                                             device_map=device_map,
                                             low_cpu_mem_usage=True)
model = torch.compile(model)

# Initialize tokenizer
tokenizer = AutoTokenizer.from_pretrained(arguments.base_model, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"


# Generate Response
device: str = "cuda" if torch.cuda.is_available() else "cpu"
generation_config = GenerationConfig(max_new_tokens=20,
                                     min_new_tokens=5,
                                     repetition_penalty=1.5,
                                     use_cache=True)
wandb.config["generation_configuration"] = generation_config.to_dict()

example_prompt = tokenizer.apply_chat_template(test_data[0]["prompt"],
                                       tokenize=False,
                                       padding=True,
                                       max_length=1024,
                                       add_generation_prompt=True,
                                       return_tensors="pt")
print(f"prompt: {example_prompt}")

test_response: list = []
for sample in tqdm(test_data, colour="green"):
    tokenized_prompt = tokenizer.apply_chat_template(sample["prompt"],
                                                     tokenize=True,
                                                     padding=True,
                                                     max_length=1024,
                                                     add_generation_prompt=True,
                                                     return_tensors="pt").to(device)
    encoded_response = model.generate(tokenized_prompt, generation_config=generation_config)
    response_raw = tokenizer.decode(encoded_response[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    response = response_raw.replace(response_raw.split("[/INST]")[0], "").removeprefix("[/INST]").strip()
    test_response.append(response)

result = test_data.add_column("test_response", test_response).remove_columns("prompt")

# text_generator = ResponseGeneratorPipeline(model=model, tokenizer=tokenizer, device=device)
#
# result = test_data.map(lambda sample: {
#     "test_response": text_generator(sample)
# }, input_columns="prompt", remove_columns="prompt", batched=True, num_proc=16)

# Sentiment Analysis
sentiment_analysis_model = AutoModelForSequenceClassification.from_pretrained(
    arguments.sentiment_analysis_model,
    quantization_config=quantization_config,
    device_map=device_map,
    low_cpu_mem_usage=True)
# sentiment_analysis_model = torch.compile(sentiment_analysis_model)

sentiment_analysis_tokenizer = AutoTokenizer.from_pretrained(arguments.sentiment_analysis_tokenizer,
                                                             trust_remote_code=True)

analyser = pipeline("sentiment-analysis",
                    model=sentiment_analysis_model,
                    tokenizer=sentiment_analysis_tokenizer,
                    device_map=device_map,
                    trust_remote_code=True)

# to prevent "The model 'OptimizedModule' is not supported for sentiment-analysis." problem
sentiment_analysis_model = torch.compile(sentiment_analysis_model)

result = result.map(lambda sample: {
    "test_response_sentiment": analyser(sample)
}, input_columns="test_response", batched=True)

result = result.map(lambda sample: {
    "test_response_sentiment": sample["label"] if sample["label"] != "joy" else "happiness"
}, input_columns="test_response_sentiment", num_proc=16)

# Metrics
emotion_label_to_id: dict = {label: index for index, label in enumerate(emotion_labels)}

sentiment_true = result.map(lambda sample: {
    "emotion_bot_id": emotion_label_to_id[sample]
}, input_columns="emotion_bot", num_proc=16).to_list()

sentiment_pred = result.map(lambda sample: {
    "test_response_sentiment_id": emotion_label_to_id[sample]
}, input_columns="test_response_sentiment", num_proc=16).to_list()

sentiment_true: torch.tensor = torch.tensor([sample["emotion_bot_id"] for sample in sentiment_true])
sentiment_pred: torch.tensor = torch.tensor([sample["test_response_sentiment_id"] for sample in sentiment_pred])

f1_score = multiclass_f1_score(sentiment_true, sentiment_pred, num_classes=len(emotion_labels), average="micro")

accuracy = multiclass_accuracy(sentiment_true, sentiment_pred, num_classes=len(emotion_labels))

wandb.log({"F1-score": f1_score, "Accuracy": accuracy})
wandb.log({"evaluation_result": wandb.Table(dataframe=result.to_pandas())})

wandb.finish()
