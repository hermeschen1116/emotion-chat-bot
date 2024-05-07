from argparse import ArgumentParser
from dataclasses import dataclass

import torch
import wandb
from datasets import load_from_disk
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
from tqdm.auto import tqdm
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification,
                          AutoTokenizer,
                          HfArgumentParser,
                          BitsAndBytesConfig,
                          TextClassificationPipeline, GenerationConfig)
from transformers.hf_argparser import HfArg

from libs.CommonConfig import CommonScriptArguments, CommonWanDBArguments, get_torch_device


@dataclass
class ScriptArguments(CommonScriptArguments):
    chat_template_file: str = HfArg(aliases="--chat-template-file", default="")


config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)

# chat_template: dict = eval(open(args.chat_template_file, "r", encoding="utf-8", closefd=True).read())

run = wandb.init(
    job_type=wandb_args.job_type,
    config=wandb_args.config,
    project=wandb_args.project,
    group=wandb_args.group,
    notes=wandb_args.notes,
    mode=wandb_args.mode,
    resume=wandb_args.resume
)
# wandb.config["chat_template"] = chat_template["template"]
# wandb.config["instruction_template"] = chat_template["instruction"]
# wandb.config["response_template"] = chat_template["response"]
# wandb.config["special_tokens"] = chat_template["special_tokens"]


# Load and Process Dataset
dataset_path = run.use_artifact(wandb.config["dataset"]).download()
dataset = load_from_disk(dataset_path)["test"]

dataset = dataset.map(lambda samples: {
    "emotion_dialog": [sample[-1]["content"]["dialog"] for sample in samples],
    "emotion_bot": [sample[-1]["content"]["emotion"] for sample in samples],
    "prompt": [sample[:-1] for sample in samples]
}, input_columns="prompt", batched=True, num_proc=16)

dataset = dataset.map(lambda samples: {
    "history": ["\n".join(
        [f"{turn['role']}({turn['content']['emotion']}): {turn['content']['dialog']}" for turn in sample]
    ) for sample in samples]
}, input_columns="prompt", batched=True, num_proc=16)

dataset = dataset.map(lambda samples: {
    "prompt": [[{
        "role": turn["role"],
        "content": turn["content"]["dialog"]
    } for turn in sample] for sample in samples]
}, input_columns="prompt", batched=True, num_proc=16)

system_prompt: list = [{"role": "system", "content": wandb.config["system_prompt"]}]
system_prompt = system_prompt if wandb.config["system_prompt"] != "" else []

dataset = dataset.map(lambda samples: {
    "prompt": [system_prompt + sample for sample in samples]
}, input_columns="prompt", batched=True, num_proc=16)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    wandb.config["tokenizer"],
    trust_remote_code=True
)
tokenizer.padding_side = "left"
tokenizer.clean_up_tokenization_spaces = True
# tokenizer.chat_template = wandb.config["chat_template"]
# tokenizer.add_special_tokens(wandb.config["special_tokens"])
tokenizer.add_special_tokens({"pad_token": "[pad]" if tokenizer.pad_token is None else tokenizer.pad_token})
wandb.config["example_prompt"] = tokenizer.apply_chat_template(dataset[0]["prompt"], tokenize=False)

# Load Model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    # run.use_model(wandb.config["fine_tuned_model"]),
    wandb.config["fine_tuned_model"],
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2",
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
model = torch.compile(model)

# Generate Response
device: str = get_torch_device()
generation_config = GenerationConfig(max_new_tokens=20,
                                     min_new_tokens=5,
                                     repetition_penalty=1.5,
                                     pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id)

test_response: list = []
for sample in tqdm(dataset, colour="yellow"):
    tokenized_prompt: torch.tensor = tokenizer.apply_chat_template(sample["prompt"],
                                                                   tokenize=True,
                                                                   padding=True,
                                                                   max_length=1024,
                                                                   add_generation_prompt=True,
                                                                   return_tensors="pt").to(device)
    generated_tokens: torch.tensor = model.generate(tokenized_prompt, generation_config=generation_config)
    encoded_response: torch.tensor = generated_tokens[0][tokenized_prompt.shape[1]:]
    response = tokenizer.decode(encoded_response,
                                skip_special_tokens=True,
                                clean_up_tokenization_spaces=True)
    test_response.append(response)

result = dataset.add_column("test_response", test_response)

# Sentiment Analysis
sentiment_analysis_model = AutoModelForSequenceClassification.from_pretrained(
    wandb.config["sentiment_analysis_model"],
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True
)

sentiment_analysis_tokenizer = AutoTokenizer.from_pretrained(
    wandb.config["sentiment_analysis_tokenizer"],
    trust_remote_code=True
)

analyser = TextClassificationPipeline(
    model=sentiment_analysis_model,
    tokenizer=sentiment_analysis_tokenizer,
    framework="pt",
    task="sentiment-analysis",
    num_workers=16,
    torch_dtype="auto"
)

# to prevent "The model 'OptimizedModule' is not supported for sentiment-analysis." problem
sentiment_analysis_model = torch.compile(sentiment_analysis_model)

result = result.add_column("test_response_sentiment", analyser(result["test_response"]))

# Metrics
emotion_labels: list = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
emotion_id: dict = {label: index for index, label in enumerate(emotion_labels)}

result = result.map(lambda samples: {
    "emotion_bot_id": [emotion_id[sample] for sample in samples["emotion_bot"]],
    "test_response_sentiment_id": [emotion_id[sample["label"]]
                                   for sample in samples["test_response_sentiment"]]
}, input_columns="emotion_bot", batched=True, num_proc=16)

sentiment_true: torch.tensor = torch.tensor([sample for sample in result["emotion_bot_id"].to_list()])
sentiment_pred: torch.tensor = torch.tensor([sample for sample in result["test_response_sentiment_id"].to_list()])
result = result.remove_columns(["emotion_bot_id", "test_response_sentiment_id"])

num_emotion_labels: int = len(emotion_labels)
wandb.log({
    "F1-score": multiclass_f1_score(sentiment_true, sentiment_pred, num_classes=num_emotion_labels, average="weighted"),
    "Accuracy": multiclass_accuracy(sentiment_true, sentiment_pred, num_classes=num_emotion_labels)})
wandb.log({"evaluation_result": wandb.Table(dataframe=result.to_pandas())})

wandb.finish()
