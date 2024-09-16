from argparse import ArgumentParser
from collections import Counter
from dataclasses import Field, dataclass

import torch
import wandb
from datasets import load_dataset
from libs import CommonScriptArguments, CommonWanDBArguments, ResponseGeneratorPipeline
from sklearn.metrics import classification_report
from torch import Tensor
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
from transformers import GenerationConfig, HfArgumentParser, TextStreamer, pipeline
from transformers.hf_argparser import HfArg
from unsloth import FastLanguageModel


@dataclass
class ScriptArguments(CommonScriptArguments):
    chat_template_file: Field[str] = HfArg(aliases="--chat-template-file", default="")


config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)

chat_template: dict = eval(
    open(args.chat_template_file, "r", encoding="utf-8", closefd=True).read()
)

run = wandb.init(
    job_type=wandb_args.job_type,
    config=wandb_args.config,
    project=wandb_args.project,
    group=wandb_args.group,
    notes=wandb_args.notes,
    mode=wandb_args.mode,
    resume=wandb_args.resume,
)
run.config["chat_template"] = chat_template["template"]
run.config["instruction_template"] = chat_template["instruction"]
run.config["response_template"] = chat_template["response"]
run.config["special_tokens"] = chat_template["special_tokens"]


# Load and Process Dataset
dataset = load_dataset(
    run.config["dataset"], split="test", num_proc=16, trust_remote_code=True
)

dataset = dataset.map(
    lambda samples: {
        "dialog_bot": [sample[-1]["content"]["dialog"] for sample in samples],
        "emotion_bot": [sample[-1]["content"]["emotion"] for sample in samples],
    },
    input_columns="prompt",
    batched=True,
    num_proc=16,
)

dataset = dataset.map(
    lambda samples: {
        "history": [
            "\n".join(
                [
                    f"{turn['role']}({turn['content']['emotion']}): {turn['content']['dialog']}"
                    for turn in sample[:-1]
                ]
            )
            for sample in samples
        ]
    },
    input_columns="prompt",
    batched=True,
    num_proc=16,
)

system_prompt: list = [
    {
        "role": "system",
        "content": {"emotion": "", "dialog": run.config["system_prompt"]},
    }
]

dataset = dataset.map(
    lambda samples: {
        "prompt": [
            system_prompt
            + sample[:-1]
            + [
                {
                    "role": "assistant",
                    "content": {
                        "emotion": sample[-1]["content"]["emotion"],
                        "dialog": "",
                    },
                }
            ]
            for sample in samples
        ]
    },
    input_columns="prompt",
    batched=True,
    num_proc=16,
)

# DPO model
###########################

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Shotaro30678/response_generator_DPO",  # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
###########################

# SFT model
###########################
# Load Tokenizer
# base_model, tokenizer = FastLanguageModel.from_pretrained(
#     run.config["base_model"],
#     attn_implementation="flash_attention_2",
#     pretraining_tp=1,
#     load_in_4bit=True,
#     device_map="auto",
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
# )
# tokenizer.padding_side = "left"
# tokenizer.clean_up_tokenization_spaces = True
# tokenizer.chat_template = run.config["chat_template"]
# tokenizer.add_special_tokens(run.config["special_tokens"])
# base_model.resize_token_embeddings(len(tokenizer))

# run.config["example_prompt"] = tokenizer.apply_chat_template(dataset[0]["prompt"], tokenize=False)

# model = PeftModel.from_pretrained(base_model, run.config["fine_tuned_model"])
# model = torch.compile(model)
# FastLanguageModel.for_inference(model)
###########################

# Generate Response
bot = ResponseGeneratorPipeline(
    model,
    tokenizer,
    framework="pt",
    task="conversation-generation",
    num_workers=16,
    torch_dtype="auto",
    add_special_tokens=True,
    truncation=False,
    padding=True,
)

streamer = TextStreamer(
    tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=True
)

generation_config = GenerationConfig(
    max_new_tokens=150,  # Reduce the max tokens to generate
    min_new_tokens=5,
    repetition_penalty=1.1,
    top_k=3,  # Reduce the top_k sampling
    top_p=0.9,  # Reduce the top_p sampling
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    temperature=1.0,  # Adjust temperature for faster response
    do_sample=True,  # Sampling instead of beam search
    num_beams=1,  # Greedy search
)

result = dataset.map(
    lambda sample: {
        "test_response": bot(
            sample,
            streamer=streamer,
            generation_config=generation_config,
            tokenizer=tokenizer,
        )[0]["generated_text"][-1]["content"]["dialog"]
    },
    input_columns="prompt",
)
result = result.remove_columns("prompt")

# Sentiment Analysis
emotion_labels: list = [
    "neutral",
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
]

sentiment_analyser = pipeline(
    model=run.config["sentiment_analyser_model"],
    framework="pt",
    task="sentiment-analysis",
    num_workers=16,
    device_map="auto",
    torch_dtype="auto",
    model_kwargs={
        "id2label": {k: v for k, v in enumerate(emotion_labels)},
        "label2id": {v: k for k, v in enumerate(emotion_labels)},
        "low_cpu_mem_usage": True,
    },
    trust_remote_code=True,
)

# Detect gibberish
gibberish_analyser = pipeline(
    model=run.config["gibberish_detector_model"],
    tokenizer=run.config["gibberish_detector_model"],
    max_length=512,
    framework="pt",
    task="text-classification",
    num_workers=16,
    device_map="cpu",
    torch_dtype="auto",
    model_kwargs={"low_cpu_mem_usage": True},
    trust_remote_code=True,
)

result = result.add_column(
    "test_response_sentiment", sentiment_analyser(result["test_response"])
)
result = result.add_column(
    "test_response_gibberish", gibberish_analyser(result["test_response"])
)

gibberish_labels = [sample["label"] for sample in result["test_response_gibberish"]]
label_distribution = Counter(gibberish_labels)

wandb.log({"Gibberish level": dict(label_distribution)})

incomplete_idx = [
    sample.rstrip(" ")[-1:] not in ["!", ".", "?"] for sample in result["test_response"]
]
incomplete_num = Counter(incomplete_idx)

wandb.log({"Incomplete amount": dict(incomplete_num)})

# Metrics
emotion_id: dict = {label: index for index, label in enumerate(emotion_labels)}

sentiment_true: Tensor = torch.tensor(
    [emotion_id[sample] for sample in result["emotion_bot"]]
)
sentiment_pred: Tensor = torch.tensor(
    [emotion_id[sample["label"]] for sample in result["test_response_sentiment"]]
)

num_emotion_labels: int = len(emotion_labels)

wandb.log(
    {
        "F1-score": multiclass_f1_score(
            sentiment_true,
            sentiment_pred,
            num_classes=num_emotion_labels,
            average="weighted",
        ),
        "Accuracy": multiclass_accuracy(
            sentiment_true, sentiment_pred, num_classes=num_emotion_labels
        ),
    }
)
wandb.log({"evaluation_result": wandb.Table(dataframe=result.to_pandas())})

wandb.finish()

print(
    classification_report(sentiment_true, sentiment_pred, target_names=emotion_labels)
)
