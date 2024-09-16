from argparse import ArgumentParser

import torch
import wandb
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch import Tensor
from torcheval.metrics.functional import multiclass_accuracy, multiclass_f1_score
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)

from models.libs.CommonConfig import CommonScriptArguments, CommonWanDBArguments

config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((CommonScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)

run = wandb.init(
    name=wandb_args.name,
    job_type=wandb_args.job_type,
    config=wandb_args.config,
    project=wandb_args.project,
    group=wandb_args.group,
    notes=wandb_args.notes,
)

dataset = load_dataset(
    run.config["dataset"],
    num_proc=16,
    trust_remote_code=True,
).remove_columns(["act"])

dataset = dataset.map(
    lambda samples: {
        "response_emotion": [sample[1:] + [sample[0]] for sample in samples]
    },
    input_columns=["emotion"],
    remove_columns=["emotion"],
    batched=True,
    num_proc=16,
)

dataset = dataset.map(
    lambda samples: {
        "dialog": [sample[:-1] for sample in samples["dialog"]],
        "response_emotion": [sample[:-1] for sample in samples["response_emotion"]],
    },
    batched=True,
    num_proc=16,
)

dataset = dataset.map(
    lambda samples: {
        "dialog": [sample[:-1] for sample in samples["dialog"]],
        "response_emotion": [sample[:-1] for sample in samples["response_emotion"]],
    },
    batched=True,
    num_proc=16,
)

dataset = dataset.map(
    lambda samples: {
        "row": [
            [
                {
                    "text": dialog,
                    "label": emotion,
                }
                for i, (emotion, dialog) in enumerate(zip(sample[0], sample[1]))
            ]
            for sample in zip(samples["response_emotion"], samples["dialog"])
        ]
    },
    remove_columns=["response_emotion", "dialog"],
    batched=True,
    num_proc=16,
)
dataset = dataset.flatten().rename_columns({"row.label": "label", "row.text": "text"})

emotion_labels: list = dataset["train"].features["label"].feature.names
emotion_labels[0] = "neutral"
num_emotion_labels: int = len(emotion_labels)

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = "right"
tokenizer.clean_up_tokenization_spaces = True

base_model = AutoModel.from_pretrained(
    run.config["base_model"],
    attn_implementation="flash_attention_2",
    num_labels=num_emotion_labels,
    id2label={k: v for k, v in enumerate(emotion_labels)},
    label2id={v: k for k, v in enumerate(emotion_labels)},
    use_cache=False,
    device_map="auto",
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

dataset = dataset.map(
    lambda samples: {
        "tokenized_text": [
            tokenizer.encode(sample, padding="max_length", truncation=True)
            for sample in samples
        ],
    },
    input_columns=["text"],
    batched=True,
    num_proc=16,
)


peft_config = LoraConfig(
    task_type="SEQ_CLS",
    lora_alpha=run.config["lora_alpha"],
    lora_dropout=run.config["lora_dropout"],
    r=run.config["lora_rank"],
    bias=run.config["lora_bias"],
    init_lora_weights=run.config["init_lora_weights"],
    use_rslora=run.config["use_rslora"],
)
base_model = get_peft_model(base_model, peft_config)


def compute_metrics(prediction) -> dict:
    sentiment_true: Tensor = torch.tensor(
        [[label] for label in prediction.label_ids.tolist()]
    )
    sentiment_pred: Tensor = torch.tensor(
        [label] for label in prediction.predictions.argmax(-1).tolist()
    )

    return {
        "Accuracy": multiclass_accuracy(
            sentiment_true, sentiment_pred, num_classes=num_emotion_labels
        ),
        "F1-score": multiclass_f1_score(
            sentiment_true,
            sentiment_pred,
            num_classes=num_emotion_labels,
            average="weighted",
        ),
    }


per_device_batch_size: int = 8
logging_steps: int = len(dataset["train"]) // per_device_batch_size
trainer_arguments = TrainingArguments(
    output_dir="./checkpoints",
    overwrite_output_dir=True,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    learning_rate=run.config["learning_rate"],
    lr_scheduler_type="constant",
    weight_decay=run.config["weight_decay"],
    max_grad_norm=run.config["max_grad_norm"],
    num_train_epochs=run.config["num_train_epochs"],
    warmup_ratio=run.config["warmup_ratio"],
    max_steps=run.config["max_steps"],
    logging_steps=logging_steps,
    log_level="error",
    save_steps=500,
    save_total_limit=2,
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    fp16=True,
    bf16=False,
    dataloader_num_workers=16,
    optim=run.config["optim"],
    group_by_length=True,
    report_to=["wandb"],
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": True},
    auto_find_batch_size=True,
    torch_compile=False,
    include_tokens_per_second=True,
    include_num_input_tokens_seen=True,
)

tunner = Trainer(
    model=base_model,
    args=trainer_arguments,
    compute_metrics=compute_metrics,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
)

tunner.train()

tunner.push_to_hub(run.config["fine_tuned_model"])

wandb.finish()
