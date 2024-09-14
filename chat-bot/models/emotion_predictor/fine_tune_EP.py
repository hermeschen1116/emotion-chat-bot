import os
import random

import torch
import wandb

from datasets import load_dataset, Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from huggingface_hub import login
from dotenv import load_dotenv
from peft import LoraConfig, get_peft_model

load_dotenv()
login(token=os.environ.get("HF_TOKEN", ""), add_to_git_credential=True)
wandb.login(key=os.environ.get("WANDB_API_KEY", ""))

wandb_config = {
    "base_model": "michellejieli/emotion_text_classifier",
}
wandb.init(
    job_type="fine-tuning",
    config=wandb_config,
    project="emotion-chat-bot-ncu-half-neutral-data",
    group="emotion_predictor_ex1",
    mode="online",
    # resume="auto"
)

base_model = "michellejieli/emotion_text_classifier"
new_model = "./etc_on_dd-half_neutral"

num_labels = 7

emotion_labels: list = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
id2label = {k: v for k, v in enumerate(emotion_labels)}
label2id = {v: k for k, v in enumerate(emotion_labels)}

def preprocessing(data):
    data = data.rename_column("utterance", "text")
    data = data.rename_column("emotion", "label")
    data = data.remove_columns("turn_type")
    return data

def shift_labels(dataset):
    df = dataset.to_pandas()
    df["label"] = df.groupby('dialog_id')["label"].shift(-1)
    df.dropna(inplace = True)
    df["label"]  = df["label"].astype(int)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.remove_columns("dialog_id")
    return dataset

def remove_half_train(data):
    data_set = data["train"]
    label_0_indices = [i for i, row in enumerate(data_set) if row['label'] == 0]
    num_to_remove = len(label_0_indices) // 2
    indices_to_remove = random.sample(label_0_indices, num_to_remove)
    filtered_data = data_set.filter(lambda x, i: i not in indices_to_remove, with_indices=True)
    data["train"] = filtered_data
    return data

def shift_all(data):
    data["train"] = shift_labels(data["train"])
    data["validation"] = shift_labels(data["validation"])
    data["test"] = shift_labels(data["test"])
    return data

data_name = "benjaminbeilharz/better_daily_dialog"
data_raw = load_dataset(data_name, num_proc=16)
data_raw = preprocessing(data_raw)
data_raw = shift_all(data_raw)
data_raw = remove_half_train(data_raw)
data = data_raw

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForSequenceClassification.from_pretrained(
    base_model,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id
    )

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

emotions = data
emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
emotions_encoded = emotions_encoded.remove_columns(['__index_level_0__']) # some sort of weird bug

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lora_config = LoraConfig(
    lora_alpha=64,
    lora_dropout=0.2,
    r=128,
    bias="none",
    task_type="SEQ_CLS",
    use_rslora = True
)
peft_model = get_peft_model(model, lora_config)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

batch_size = 8
logging_steps = len(emotions_encoded["train"]) // batch_size

training_args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=5,
    load_best_model_at_end = True,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_steps=500,
    save_total_limit=2,
    save_strategy = "epoch",
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
    evaluation_strategy="epoch",
    log_level="error",
    overwrite_output_dir=True
)
wandb.config["trainer_arguments"] = training_args.to_dict()

trainer = Trainer(model=peft_model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)
trainer.train()
wandb.finish()

trainer.model.save_pretrained(new_model)