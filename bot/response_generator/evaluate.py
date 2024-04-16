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
from transformers.utils.hub import move_cache

import wandb
# from Pipeline import ResponseGeneratorPipeline

# commandline inputs
parser = argparse.ArgumentParser(prog="Evaluation", description="Evaluation Script For Response Generator")
parser.add_argument("--tokenizer", required=True, type=str)
parser.add_argument("--fine_tuned_model", required=False, type=str, default="")
parser.add_argument("--sentiment_analysis_tokenizer",
                    required=False,
                    type=str,
                    default="michellejieli/emotion_text_classifier")
parser.add_argument("--sentiment_analysis_model",
                    required=False,
                    type=str,
                    default="michellejieli/emotion_text_classifier")
parser.add_argument("--experiment_detail", required=True, type=str, default="")
parser.add_argument(
    "--wandb_mode",
    required=False,
    type=str,
    choices=["online", "offline", "disabled"],
    default="online"
)
parser.add_argument("--enable_flash_attention_2", required=False, type=bool, default=True)
parser.add_argument(
    "--system_prompt_mode",
    required=False,
    type=str,
    choices=["default", "custom", "disabled"],
    default="disabled"
)
parser.add_argument("--system_prompt", required=False, type=str, default=None)
parser.add_argument("--chat_template_file", required=True, type=str, default="")

arguments = parser.parse_args()
arguments.fine_tuned_model = arguments.tokenizer if arguments.fine_tuned_model == "" else arguments.fine_tuned_model
if arguments.system_prompt_mode == "disabled":
    arguments.system_prompt = None
else:
    arguments.system_prompt = "" if arguments.system_prompt_mode == "default" else arguments.system_prompt
chat_template: dict = eval(open(arguments.chat_template_file, "r", encoding="utf-8", closefd=True).read())

# prevent env load failed
load_dotenv(encoding="utf-8")
huggingface_hub.login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=True)
wandb.login(key=os.environ.get("WANDB_API_KEY", ""), relogin=True)

print(f"tokenizer: {arguments.tokenizer}, fine-tuned model: {arguments.fine_tuned_model}")
# Initialize Wandb
wandb_config: dict = {
    "tokenizer": arguments.tokenizer,
    "fine_tuned_model": arguments.fine_tuned_model,
    "system_prompt": arguments.system_prompt,
    "chat_template": chat_template["template"],
    "instruction_template": chat_template["instruction"],
    "response_template": chat_template["response"],
    "additional_special_tokens": chat_template["special_tokens"]
}
run = wandb.init(
    job_type="evaluation",
    config=wandb_config,
    project="emotion-chat-bot-ncu",
    group="Response Generator",
    notes=arguments.experiment_detail,
    mode=arguments.wandb_mode,
    allow_val_change=True
)

# Load and Process Dataset
dataset = load_dataset("daily_dialog",
                       split="train+validation",
                       num_proc=16,
                       trust_remote_code=True).remove_columns("act")
dataset = dataset.rename_column("emotion", "emotion_id")
emotion_labels: list = dataset.features["emotion_id"].feature.names
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
dataset = dataset.map(lambda samples: {
    "emotion_history": [sample[:-1] for sample in samples],
    "emotion_bot": [sample[-1] for sample in samples]
}, input_columns="emotion", remove_columns="emotion", batched=True, num_proc=16)
dataset = dataset.map(lambda samples: {
    "dialog_history": [sample[:-1] for sample in samples],
    "dialog_bot": [sample[-1] for sample in samples]
}, input_columns="dialog", remove_columns="dialog", batched=True, num_proc=16)
test_data = dataset.map(lambda samples: {
    "prompt": [[{
        "role": "user" if i % 2 == 0 else "assistant",
        "content": {"emotion": emotion, "dialog": dialog}
    }
        for i, (emotion, dialog) in enumerate(zip(sample[0], sample[1]))]
        for sample in zip(samples["emotion_history"], samples["dialog_history"])]
}, batched=True, num_proc=16)
test_data = test_data.map(lambda sample: {
    "history": "\n".join([f"""{'user' if i % 2 == 0 else 'bot'}({v[0]}): {v[1]}"""
                          for i, v in enumerate(zip(sample["emotion_history"], sample["dialog_history"]))])
}, remove_columns=["emotion_history", "dialog_history"], num_proc=8)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(arguments.tokenizer, trust_remote_code=True)
tokenizer.padding_side = "left"
tokenizer.clean_up_tokenization_spaces = True
tokenizer.chat_template = chat_template["template"]
tokenizer.add_special_tokens(chat_template["special_tokens"], replace_additional_special_tokens=True)


def prompt_compose(sample: str) -> str:
    return tokenizer.apply_chat_template(sample,
                                         tokenize=False,
                                         padding=True,
                                         max_length=4096,
                                         return_tensors="pt"
                                         )


test_data = test_data.map(lambda sample: {
    "prompt": prompt_compose(sample)
}, input_columns="prompt", num_proc=16)
wandb.config["example_prompt"] = test_data[0]["prompt"]

# Load Model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
quantization_config = quantization_config if torch.cuda.is_available() else None
wandb.config["quantization_configuration"] = quantization_config.to_dict() if quantization_config is not None else {}

flash_attention: str = "flash_attention_2" if arguments.enable_flash_attention_2 else None
model = AutoModelForCausalLM.from_pretrained(
    run.use_model(arguments.fine_tuned_model),
    quantization_config=quantization_config,
    attn_implementation=flash_attention,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
model.resize_token_embeddings(len(tokenizer))
wandb.config["model_configuration"] = model.config.to_dict()
model = torch.compile(model)


# Generate Response
device: str = "cuda" if torch.cuda.is_available() else "cpu"
generation_config = GenerationConfig(
    max_new_tokens=20,
    min_new_tokens=5,
    repetition_penalty=1.5,
    use_cache=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)
wandb.config["generation_configuration"] = generation_config.to_dict()

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
    device_map="auto",
    low_cpu_mem_usage=True)
# sentiment_analysis_model = torch.compile(sentiment_analysis_model)

sentiment_analysis_tokenizer = AutoTokenizer.from_pretrained(arguments.sentiment_analysis_tokenizer,
                                                             trust_remote_code=True)

analyser = pipeline("sentiment-analysis",
                    model=sentiment_analysis_model,
                    tokenizer=sentiment_analysis_tokenizer,
                    device_map="auto",
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

num_emotion_labels: int = len(emotion_labels)
wandb.log({
    "F1-score": multiclass_f1_score(sentiment_true, sentiment_pred, num_classes=num_emotion_labels, average="micro"),
    "Accuracy": multiclass_accuracy(sentiment_true, sentiment_pred, num_classes=num_emotion_labels)
})
wandb.log({"evaluation_result": wandb.Table(dataframe=result.to_pandas())})

wandb.finish()
