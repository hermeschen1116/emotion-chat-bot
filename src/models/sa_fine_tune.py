import random
from argparse import ArgumentParser
from dataclasses import Field, dataclass

import torch
import wandb
from datasets import load_dataset
from libs import CommonScriptArguments, CommonWanDBArguments
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.hf_argparser import HfArg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class ScriptArguments(CommonScriptArguments):
    chat_template_file: Field[str] = HfArg(aliases="--chat-template-file", default="")


config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)

# Initialize Wandb
run = wandb.init(
    job_type=wandb_args.job_type,
    config=wandb_args.config,
    project=wandb_args.project,
    group=wandb_args.group,
    notes=wandb_args.notes,
    mode=wandb_args.mode,
    resume=wandb_args.resume,
)


def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def remove_half_neutral(data):
    data_set = data["train"]
    label_0_indices = [i for i, row in enumerate(data_set) if row["label"] == 0]
    num_to_remove = len(label_0_indices) // 2
    indices_to_remove = random.sample(label_0_indices, num_to_remove)
    filtered_data = data_set.filter(
        lambda x, i: i not in indices_to_remove, with_indices=True
    )
    data["train"] = filtered_data
    return data

num_labels = 7

emotion_labels: list = [
    "neutral",
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
]
id2label = {k: v for k, v in enumerate(emotion_labels)}
label2id = {v: k for k, v in enumerate(emotion_labels)}


def preprocessing(data):
    data = data.rename_column("utterance", "text")
    data = data.rename_column("emotion", "label")
    data = data.remove_columns(["dialog_id", "turn_type"])
    return data


raw_data = load_dataset(
    wandb.config["dataset"],
    num_proc=16
    )

processed_data = preprocessing(raw_data)
half_neutral_data = remove_half_neutral(processed_data)

base_model = wandb.config["base_model"]

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForSequenceClassification.from_pretrained(
    base_model, num_labels=num_labels, id2label=id2label, label2id=label2id
)

encoded_data = half_neutral_data.map(tokenize, batched=True, batch_size=None)

lora_config = LoraConfig(
    lora_alpha=64,
    lora_dropout=0.2,
    r=128,
    bias="none",
    task_type="SEQ_CLS",
    use_rslora=True,
)
peft_model = get_peft_model(model, lora_config)

batch_size = 8
logging_steps = len(encoded_data["train"]) // batch_size

training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=2,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=5000,
    save_total_limit=2,
    save_strategy="epoch",
    logging_steps=logging_steps,
    learning_rate=0.0001,
    weight_decay=0.1,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to=["wandb"],
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    eval_strategy="epoch",
    overwrite_output_dir=True,
)
wandb.config["trainer_arguments"] = training_args.to_dict()

trainer = Trainer(
    model=peft_model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=encoded_data["train"],
    eval_dataset=encoded_data["validation"],
    tokenizer=tokenizer,
)

trainer.train()
trainer.model.save_pretrained("sentiment_analyzer_DD")
