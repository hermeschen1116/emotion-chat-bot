from argparse import ArgumentParser
from dataclasses import Field, dataclass

import wandb

from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import pipeline
from libs import CommonScriptArguments, CommonWanDBArguments
from transformers import HfArgumentParser
from transformers.hf_argparser import HfArg

@dataclass
class ScriptArguments(CommonScriptArguments):
    chat_template_file: Field[str] = HfArg(aliases="--chat-template-file", default="")


config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((ScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)

run = wandb.init(
    job_type=wandb_args.job_type,
    config=wandb_args.config,
    project=wandb_args.project,
    group=wandb_args.group,
    notes=wandb_args.notes,
    mode=wandb_args.mode,
    resume=wandb_args.resume,
)

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

dataset = load_dataset(
	wandb.config["dataset"],
	split="test",
	keep_in_memory=True,
	num_proc=16,
	trust_remote_code=True
)

analyzer = pipeline(
    "sentiment-analysis",
    model=wandb.config["fine_tuned_model"],
    device='cuda'
    )

raw_analyzer = pipeline(
    "sentiment-analysis",
    model=wandb.config["base_model"],
    device='cuda'
    )

sent_pred = []
sent_true = []
raw_sent_pred = []
        
for data in dataset:
    sent_true.append(id2label[data['emotion']])
    sent_pred.append(analyzer(data['utterance'])[0]['label'])
    
    raw_res = raw_analyzer(data['utterance'])[0]['label']
    if raw_res == 'joy':
        raw_sent_pred.append('happiness')
        continue
    else:
        raw_sent_pred.append(raw_res)
        
print(f"fine_tuned:\n{classification_report(sent_true, sent_pred)}\nbaseline:\n{classification_report(sent_true, raw_sent_pred)}")