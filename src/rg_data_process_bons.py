from argparse import ArgumentParser
from dataclasses import Field, dataclass

import numpy as np
import torch
import wandb
from datasets import Dataset, load_dataset
from peft.peft_model import PeftModel
from torch import tensor
from tqdm.auto import tqdm
from transformers import HfArgumentParser, TextStreamer, pipeline
from transformers.hf_argparser import HfArg
from unsloth import FastLanguageModel

from libs import CommonScriptArguments, CommonWanDBArguments, ResponseGeneratorPipeline


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
wandb.config["chat_template"] = chat_template["template"]
wandb.config["instruction_template"] = chat_template["instruction"]
wandb.config["response_template"] = chat_template["response"]
wandb.config["special_tokens"] = chat_template["special_tokens"]

# Load Dataset
dataset = load_dataset(
    wandb.config["dataset"],
    split="train+validation",
    keep_in_memory=True,
    num_proc=16,
    trust_remote_code=True,
)

# dataset filtering
history_length: int = 2 * wandb.config["num_turns_history"]
dataset = dataset.filter(
    lambda sample: len(sample) >= (2 + history_length),
    input_columns="prompt",
    num_proc=16,
)
print(f"dataset size after filter: {len(dataset)}")

# take certain amount of data to see if it works
dataset = dataset.take(2048)

# dataset preprocessing
dataset = dataset.map(
    lambda sample: {
        "prompt": sample[i : i + 2 + history_length]
        for i in range(0, len(sample) - 2, 2)
        if (i + 2 + history_length) <= len(sample)
    },
    input_columns="prompt",
    num_proc=16,
)

system_prompt: list = [
    {
        "role": "system",
        "content": {"emotion": "", "dialog": wandb.config["system_prompt"]},
    }
]

dataset = dataset.map(
    lambda samples: {"prompt": [system_prompt + sample for sample in samples]},
    input_columns="prompt",
    batched=True,
    num_proc=16,
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

dataset = dataset.map(
    lambda samples: {
        "query": [
            sample[:-1]
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
        ],
        "label": [sample[-1]["content"]["emotion"] for sample in samples],
    },
    input_columns="prompt",
    batched=True,
    num_proc=16,
)

# Target difference of chosen and rejected
target_score_range = wandb.config["target_score_range"]

# Load Tokenizer
base_model, tokenizer = FastLanguageModel.from_pretrained(
    wandb.config["base_model"],
    attn_implementation="flash_attention_2",
    pretraining_tp=1,
    load_in_4bit=True,
    use_cache=False,
    device_map="auto",
    use_gradient_checkpointing=True,
    low_cpu_mem_usage=True,
)
tokenizer.padding_side = "left"
tokenizer.clean_up_tokenization_spaces = True
tokenizer.chat_template = wandb.config["chat_template"]
tokenizer.add_special_tokens(wandb.config["special_tokens"])
base_model.resize_token_embeddings(len(tokenizer))

base_model_with_adapter = PeftModel.from_pretrained(base_model, wandb.config["adapter"])
base_model_with_adapter.print_trainable_parameters()
FastLanguageModel.for_inference(base_model_with_adapter)

# dataset to torch format and etc.
dataset = dataset.with_format("torch")
dataset = dataset.map(
    lambda sample: {
        "input_ids": tokenizer.apply_chat_template(
            sample,
            tokenize=True,
            padding="max_length",
            max_length=wandb.config["max_input_tokens"],
            add_generation_prompt=True,
            return_tensors="pt",
        )
    },
    input_columns="query",
    num_proc=16,
)

print(dataset)

# Text generation
bot = ResponseGeneratorPipeline(
    base_model_with_adapter,
    tokenizer,
    framework="pt",
    task="conversation-generation",
    num_workers=5,  # cause some issue here
    torch_dtype="auto",
    add_special_tokens=True,
    truncation=False,
    padding=True,
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

# Sentiment Analysis
sentiment_analyser = pipeline(
    model=wandb.config["sentiment_analysis_model"],
    tokenizer=wandb.config["sentiment_analysis_model"],
    max_length=512,
    framework="pt",
    task="sentiment-analysis",
    num_workers=16,
    device_map="cpu",
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
    model=wandb.config["gibberish_detector_model"],
    tokenizer=wandb.config["gibberish_detector_model"],
    max_length=512,
    framework="pt",
    task="text-classification",
    num_workers=16,
    device_map="cpu",
    torch_dtype="auto",
    model_kwargs={"low_cpu_mem_usage": True},
    trust_remote_code=True,
)


def emotion_reward(response: str, emotion: str) -> float:
    score = sentiment_analyser(response)[0]

    if score["label"] == emotion:
        return score["score"] * 10
    else:
        return score["score"] * 0


def non_gibberish_reward(response: str) -> float:
    score = gibberish_analyser(response)[0]

    match score["label"]:
        case "clean":
            return score["score"] * 10
        case "mild gibberish":
            return score["score"] * 0.5
        case _:
            return score["score"] - 2


# [TODO] 用級距的方式來給予分數
def length_reward(response_length: int) -> float:
    difference_ratio_min = (response_length - 5) / 5
    difference_ratio_max = (response_length - 20) / 20

    if abs(difference_ratio_min) < 1:
        return difference_ratio_min * 0.0001
    elif abs(difference_ratio_min) > 1 > abs(difference_ratio_max):
        return abs(difference_ratio_min + difference_ratio_max) * 10
    else:
        return difference_ratio_max * 0.9


def reward(batch: dict) -> list:
    emotion_scores = [
        emotion_reward(response, batch["label"]) for response in batch["response"]
    ]
    length_scores = [
        length_reward(response_length) for response_length in batch["response_length"]
    ]
    gibberish_scores = [
        non_gibberish_reward(response) for response in batch["response"]
    ]

    # print("\nemotion_scores: ",emotion_scores)
    # print("\ngibberish_scores: ",gibberish_scores)
    # print("\nlength_scores: ",length_scores)

    reward_weight = tensor(wandb.config["reward_weights"], dtype=torch.float)
    reward_bias = tensor(wandb.config["reward_bias"], dtype=torch.float)
    return [
        reward_weight.dot(tensor(reward_score, dtype=torch.float)) + reward_bias
        for reward_score in zip(emotion_scores, length_scores, gibberish_scores)
    ]


streamer = TextStreamer(
    tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=True
)

gen_kwargs = {
    "max_new_tokens": wandb.config["max_new_tokens"],
    "min_new_tokens": wandb.config["min_new_tokens"],
    "repetition_penalty": wandb.config["repetition_penalty"],
    "top_k": 5,
    "top_p": 1.0,
    "temperature": 2.0,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id,
    "do_sample": "False",
}

# set best_of candidates amount
N_BEST_OF = wandb.config["n_best_of"]

# :: [Resp]
response_tensors_ref, response_tensors = [], []
response_tensors_ref_len, response_tensors_len = [], []
# :: [[Resp]]
response_tensors_best_of = []
response_tensors_best_of_len = []
input_ref = []


def calculate_score_diff(chosen_score, rejected_score):
    return chosen_score - rejected_score


query_tensors = [input_ids.squeeze(0) for input_ids in dataset["input_ids"]]

# Create dict for dataset
keys = ["prompt", "chosen", "rejected", "chosen_score", "rejected_score"]
tmp_data = {key: [None] * len(dataset) for key in keys}

# list of index of generation failure
fail_index = []

for i in tqdm(range(len(dataset))):
    data = dataset[i]
    query = query_tensors[i]
    input_text = tokenizer.decode(query, skip_special_tokens=True)

    input_len = len(input_text)
    input_ref.append(input_text)

    inputs = [input_text for _ in range(N_BEST_OF)]

    # prevent endless generation
    fail_counter = 0

    while True:
        output = bot(inputs, **gen_kwargs)
        responses = [text[0]["generated_text"][input_len:] for text in output]
        tmp = {
            "response": responses,
            "response_length": [len(response) for response in responses],
            "label": data["label"],
        }
        score_tmp = [
            reward.item() for reward in reward(tmp)
        ]  # Use item() to get Python scalar
        tmp["score"] = score_tmp
        score_range = calculate_score_diff(max(score_tmp), min(score_tmp))

        # If the generated output score range is less than expected, regenerate
        print(f"\nRange of scores: {score_range:.3f}")

        if score_range < target_score_range or max(score_tmp) < 8:
            fail_counter += 1
            print(f"fail: {fail_counter}/{30}")
            if fail_counter <= 30:
                print("\nRegenerating...")
                continue
            else:
                fail_index.append(i)
                break

        # # Print out some info for reference
        # print(f"\nLabel: {tmp['label']}\n")
        # print(f"assistant: {dataset[i]['query'][4]['content']['dialog']}")
        # print(f"user: {dataset[i]['query'][5]['content']['dialog']}\n")
        # # Print output
        # for j in range(N_BEST_OF):
        #     print(f"Score {j}: {score_tmp[j]:.3f}, Response: {tmp['response'][j]} \n {sentiment_analyser(tmp['response'][j])[0]}")
        chosen = tmp["response"][score_tmp.index(max(score_tmp))]
        rejected = tmp["response"][score_tmp.index(min(score_tmp))]

        chosen_sentiment = sentiment_analyser(chosen)[0]
        chosen_gibberish = gibberish_analyser(chosen)[0]

        if (
            chosen_sentiment["label"] != tmp["label"]
            or chosen_gibberish["label"] != "clean"
            or chosen_gibberish["score"] < 0.8
            or chosen.strip(" ")[-1:] not in ["!", ".", "?"]
            # add this line to force "chosen" end with ["!", ".", "?"]
        ):
            fail_counter += 1
            print(f"\nfail: {fail_counter}/{30}")
            if fail_counter <= 30:
                print("\nRegenerating...")
                continue
            else:
                fail_index.append(i)
                break

        print(f"\nchosen : {chosen}")
        print(f"rejected : {rejected}")

        tmp_data["prompt"][i] = input_text
        tmp_data["chosen"][i] = tmp["response"][score_tmp.index(max(score_tmp))]
        tmp_data["rejected"][i] = tmp["response"][score_tmp.index(min(score_tmp))]
        tmp_data["chosen_score"][i] = max(score_tmp)
        tmp_data["rejected_score"][i] = min(score_tmp)

        break

for i in sorted(fail_index, reverse=True):
    for key in keys:
        del tmp_data[key][i]

# Show median and mean
final_scores = [
    calculate_score_diff(chosen, rejected)
    for chosen, rejected in zip(tmp_data["chosen_score"], tmp_data["rejected_score"])
]
final_median = np.median(final_scores)
final_mean = np.mean(final_scores)
# print(f"\nOriginal Median: {original_median:.3f}, Original Mean: {original_mean:.3f}")
print(f"Final Median: {final_median:.3f}, Final Mean: {final_mean:.3f}")

# Convert updated_data back to dataset format
output_dataset = Dataset.from_dict(tmp_data)
output_dataset.push_to_hub("Shotaro30678/rlhf-RG-trl-style-v3")
