import os
from dataclasses import dataclass

import huggingface_hub
import torch
import wandb
from datasets import load_from_disk
from dotenv import load_dotenv
from torcheval.metrics.functional import (multiclass_accuracy,
                                          multiclass_f1_score)
from tqdm.auto import tqdm
from transformers import (AutoModelForCausalLM,
                          AutoModelForSequenceClassification,
                          AutoTokenizer,
                          BitsAndBytesConfig,
                          GenerationConfig,
                          pipeline, HfArgumentParser)

from libs.Config import WanDBArguments


# from Pipeline import ResponseGeneratorPipeline


@dataclass
class ScriptArguments:
    huggingface_api_token: str = ""
    wandb_api_token: str = ""
    chat_template_file: str = None


# commandline inputs
parser = HfArgumentParser((ScriptArguments, WanDBArguments, BitsAndBytesConfig, GenerationConfig))
args, wandb_args, quantization_config, generation_config = parser.parse_args()

chat_template: dict = eval(open(args.chat_template_file, "r", encoding="utf-8", closefd=True).read())
# parser.add_argument("--tokenizer", required=True, type=str)
# parser.add_argument("--fine_tuned_model", required=False, type=str, default="")
# parser.add_argument("--sentiment_analysis_tokenizer",
#                     required=False,
#                     type=str,
#                     default="michellejieli/emotion_text_classifier")
# parser.add_argument("--sentiment_analysis_model",
#                     required=False,
#                     type=str,
#                     default="michellejieli/emotion_text_classifier")
# parser.add_argument("--system_prompt", required=False, type=str, default=None)

# prevent env load failed
load_dotenv(encoding="utf-8")
huggingface_hub.login(token=os.environ.get("HF_TOKEN", args.huggingface_api_token), add_to_git_credential=True)
wandb.login(key=os.environ.get("WANDB_API_KEY", args.wandb_api_token), relogin=True)

# Initialize Wandb
run = wandb.init(wandb_args)
wandb.config["chat_template"] = wandb.config["template"]
wandb.config["instruction_template"] = chat_template["instruction"]
wandb.config["response_template"] = chat_template["response"]
wandb.config["special_tokens"] = wandb.config["special_tokens"]
# wandb_config: dict = {
#     "tokenizer": wandb_args.config["tokenizer"],
#     "fine_tuned_model": wandb_args.config["fine_tuned_model"],
#     "system_prompt": arguments.system_prompt,
#     "chat_template": wandb.config["template"],
#     "instruction_template": chat_template["instruction"],
#     "response_template": chat_template["response"],
#     "additional_special_tokens": wandb.config["special_tokens"]
# }
# run = wandb.init(
#     job_type="evaluation",
#     config=wandb_config,
#     project="emotion-chat-bot-ncu",
#     group="Response Generator",
#     notes=arguments.experiment_detail,
#     mode=arguments.wandb_mode,
#     allow_val_change=True
# )

# Load and Process Dataset
dataset_path = run.use_artifact("daily_dialog_for_RG_test:latest").download()
dataset = load_from_disk(dataset_path)

# Load Tokenizer
tokenizer = AutoTokenizer.from_pretrained(wandb_args.config["tokenizer"], trust_remote_code=True)
tokenizer.padding_side = "left"
tokenizer.clean_up_tokenization_spaces = True
tokenizer.chat_template = wandb.config["template"]
tokenizer.add_special_tokens(wandb.config["special_tokens"], replace_additional_special_tokens=True)

test_data = dataset.map(lambda sample: {
    "prompt": tokenizer.apply_chat_template(sample,
                                            tokenize=False,
                                            padding=True,
                                            max_length=4096,
                                            return_tensors="pt"
                                            )
}, input_columns="prompt", num_proc=16)
wandb.config["example_prompt"] = test_data[0]["prompt"]

# Load Model
quantization_config = quantization_config if torch.cuda.is_available() else None

model = AutoModelForCausalLM.from_pretrained(
    run.use_model(wandb_args.config["fine_tuned_model"]),
    quantization_config=quantization_config,
    attn_implementation="flash_attention_2",
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True
)
model.resize_token_embeddings(len(tokenizer))
wandb.config["model_configuration"] = model.config.to_dict()
model = torch.compile(model)

# Generate Response
device: str = "cuda" if torch.cuda.is_available() else "cpu"
# generation_config = GenerationConfig(
#     max_new_tokens=20,
#     min_new_tokens=5,
#     repetition_penalty=1.5,
#     use_cache=True,
#     pad_token_id=tokenizer.pad_token_id,
#     eos_token_id=tokenizer.eos_token_id
# )

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
    wandb_args.config["sentiment_analysis_model"],
    quantization_config=quantization_config,
    device_map="auto",
    low_cpu_mem_usage=True)
# sentiment_analysis_model = torch.compile(sentiment_analysis_model)

sentiment_analysis_tokenizer = AutoTokenizer.from_pretrained(wandb_args.config["sentiment_analysis_tokenizer"],
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
emotion_labels: list = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
emotion_label_to_id: dict = {label: index for index, label in enumerate(emotion_labels)}

sentiment_true = result.map(lambda samples: {
    "emotion_bot_id": [emotion_label_to_id[sample] for sample in samples]
}, input_columns="emotion_bot", batched=True, num_proc=16).to_list()

sentiment_pred = result.map(lambda samples: {
    "test_response_sentiment_id": [emotion_label_to_id[sample] for sample in samples]
}, input_columns="test_response_sentiment", batched=True, num_proc=16).to_list()

sentiment_true: torch.tensor = torch.tensor([sample["emotion_bot_id"] for sample in sentiment_true])
sentiment_pred: torch.tensor = torch.tensor([sample["test_response_sentiment_id"] for sample in sentiment_pred])

num_emotion_labels: int = len(emotion_labels)
wandb.log({
    "F1-score": multiclass_f1_score(sentiment_true, sentiment_pred, num_classes=num_emotion_labels, average="micro"),
    "Accuracy": multiclass_accuracy(sentiment_true, sentiment_pred, num_classes=num_emotion_labels)
})
wandb.log({"evaluation_result": wandb.Table(dataframe=result.to_pandas())})

wandb.finish()
