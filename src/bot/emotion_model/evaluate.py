from dataclasses import dataclass
from typing import Any, Optional

import torch
import wandb
from datasets import load_from_disk
from torcheval.metrics.functional import multiclass_f1_score, multiclass_accuracy
from transformers.hf_argparser import HfArgumentParser, HfArg

from libs.CommonConfig import CommonWanDBArguments, CommonScriptArguments, get_torch_device
from libs.EmotionModel import EmotionModel


@dataclass
class ScriptArguments(CommonScriptArguments):
    dtype: Optional[Any] = HfArg(aliases="--dtype", default=torch.float32)
    device: Optional[str] = HfArg(aliases="--device", default_factory=get_torch_device)


parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments))
args, wandb_args, remain_args = parser.parse_args()

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

dataset_path = run.use_artifact("daily_dialog_for_EM:latest").download()
eval_dataset = load_from_disk(dataset_path)["test"]
_, eval_dataset = eval_dataset.train_test_split(test_size=0.1)

model = EmotionModel(wandb_args.config["attention_type"], dtype=args.dtype, device=args.device)

eval_dataset = eval_dataset.map(lambda samples: {
    "user_representation": [model.representation_evolution(sample[0], sample[1])
                            for sample in zip(samples["user_representation"],
                                              samples["bot_dialog_emotion_composition"])]
}, batched=True)

eval_dataset = eval_dataset.map(lambda samples: {
    "user_most_possible_emotion": [[torch.argmax(representation) for representation in sample]
                                   for sample in samples]
}, input_columns="user_representation", batched=True, num_proc=16)

emotion_pred: torch.tensor = torch.tensor([emotion for sample in eval_dataset["user_most_possible_emotion"]
                                           for emotion in sample[1:]])
emotion_true: torch.tensor = torch.tensor([emotion for sample in eval_dataset["user_emotion"]
                                           for emotion in sample[1:]])

wandb.log({
    "F1-score": multiclass_f1_score(emotion_true, emotion_pred, num_classes=7, average="micro"),
    "Accuracy": multiclass_accuracy(emotion_true, emotion_pred, num_classes=7)
})

emotion_labels: list = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]
result = eval_dataset.map(lambda samples: {
    "user_most_possible_emotion": [[emotion_labels[emotion_id] for emotion_id in sample]
                                   for sample in samples["user_most_possible_emotion"]],
    "user_emotion": [[emotion_labels[emotion_id] for emotion_id in sample] for sample in samples["user_emotion"]]
}, batched=True, num_proc=16)

wandb.log({"evaluation_result": wandb.Table(dataframe=result.to_pandas())})

wandb.finish()
