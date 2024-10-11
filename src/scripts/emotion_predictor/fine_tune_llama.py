from argparse import ArgumentParser

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser

import wandb
from emotion_chat_bot.utils.CommonConfig import CommonScriptArguments, CommonWanDBArguments
from emotion_chat_bot.utils.CommonUtils import login_to_service

login_to_service()

config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((CommonScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)

run = wandb.init(config=wandb_args.config, project=wandb_args.project, group=wandb_args.group)

dataset = load_dataset("Shotaro30678/daily_dialog_for_EP", split="test", num_proc=16, trust_remote_code=True)
emotion_labels: list = dataset.features["label"].names
num_emotion_labels: int = len(emotion_labels)

quantization_config = BitsAndBytesConfig(
	load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(run.config["base_model"], add_prefix_space=True)
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForSequenceClassification.from_pretrained(
	run.config["base_model"],
	num_labels=num_emotion_labels,
	id2label={k: v for k, v in enumerate(emotion_labels)},
	label2id={v: k for k, v in enumerate(emotion_labels)},
	use_cache=False,
	device_map="auto",
	low_cpu_mem_usage=True,
	trust_remote_code=True,
)

peft_config = LoraConfig(
	r=run.config["lora_rank"],
	lora_alpha=run.config["lora_alpha"],
	target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
	lora_dropout=0.1,
	bias="none",
	task_type="SEQ_CLS",
	init_lora_weights=run.config["init_lora_weights"],
	use_rslora=run.config["use_rslora"],
)

base_model = prepare_model_for_kbit_training(base_model)
base_model = get_peft_model(base_model, peft_config)

base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

base_model.print_trainable_parameters()

batch_size = 8

all_outputs = []
pred = []

for i in range(0, len(dataset), batch_size):
	batch_sentences = dataset["text"][i : i + batch_size]

	inputs = tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True, max_length=512)

	inputs = {k: v.to("cuda" if torch.cuda.is_available() else "cpu") for k, v in inputs.items()}

	with torch.no_grad():
		outputs = base_model(**inputs)
		all_outputs.append(outputs["logits"])

final_outputs = torch.cat(all_outputs, dim=0)
pred = final_outputs.argmax(axis=1).cpu().numpy().tolist()


def get_metrics_result(dataset, pred):
	y_test = dataset["label"]
	y_pred = pred
	labels = [0, 1, 2, 3, 4, 5, 6]

	print("Classification Report:")
	print(classification_report(y_test, y_pred, target_names=emotion_labels, labels=labels, zero_division=0))

	print("Balanced Accuracy Score:", balanced_accuracy_score(y_test, y_pred))
	print("Accuracy Score:", accuracy_score(y_test, y_pred))


get_metrics_result(dataset, pred)
