from argparse import ArgumentParser
from dataclasses import dataclass, Field

import torch
from datasets import load_dataset
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          Trainer,
                          TrainingArguments
                         )

from transformers import HfArgumentParser
from transformers.hf_argparser import HfArg

from libs import CommonScriptArguments, CommonWanDBArguments
from sklearn.metrics import accuracy_score, f1_score

import wandb

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
	resume=wandb_args.resume
)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

num_labels = 7


emotion_labels: list = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
id2label = {k: v for k, v in enumerate(emotion_labels)}
label2id = {v: k for k, v in enumerate(emotion_labels)}

def preprocessing(data):
    data = data.rename_column("utterance", "text")
    data = data.rename_column("emotion", "label")
    data = data.remove_columns(["dialog_id", "turn_type"])
    return data

data = load_dataset(
    wandb.config['dataset'],
    num_proc=16
)

data = preprocessing(data)

base_model = wandb.config['base_model']

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForSequenceClassification.from_pretrained(
    base_model,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
)

emotions = data
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

batch_size = 64
logging_steps = len(emotions_encoded["train"]) // batch_size

training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=2,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=25,
    logging_steps=logging_steps,
    learning_rate=2e-5,
    weight_decay=0.01,
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
    log_level="error"
)

trainer = Trainer(model=model,
                  args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)

trainer.train();
trainer.model.save_pretrained("sentiment_analyzer_DD")