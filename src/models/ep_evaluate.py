from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

config_getter = ArgumentParser()
config_getter.add_argument("--json_file", required=True, type=str)
config = config_getter.parse_args()

parser = HfArgumentParser((CommonScriptArguments, CommonWanDBArguments))
args, wandb_args = parser.parse_json_file(config.json_file)


def create_classifier_pipeline(model_name, num_labels, id2label, label2id, tokenizer):
    classifier_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels, id2label=id2label, label2id=label2id
    )
    classifier = pipeline(
        "sentiment-analysis", model=classifier_model, tokenizer=tokenizer, device=0
    )
    return classifier


def evaluate(predictions):
    true_labels = [p["true_label"] for p in predictions]
    predicted_labels = [p["predicted_label"] for p in predictions]
    print(
        classification_report(
            true_labels,
            predicted_labels,
            target_names=[id2label[i] for i in range(num_labels)],
            digits=4,
        )
    )
    first10 = predicted_labels[0:9]
    f1 = f1_score(true_labels, predicted_labels, average="weighted")
    acc = accuracy_score(true_labels, predicted_labels)
    return first10, f1, acc


# full dataset 10 epoches
classifier = create_classifier_pipeline(
    new_model, num_labels, id2label, label2id, tokenizer
)
predictions = data.map(lambda row: predict(row, classifier, label2id))
print("Fine-tuned:")
ft_10, f1_ft, accuracy_ft = evaluate(predictions)

# half dataset 5 epoches
classifier = create_classifier_pipeline(
    new_model_half, num_labels, id2label, label2id, tokenizer
)
predictions = data.map(lambda row: predict(row, classifier, label2id))
print("\nFine-tuned-half:")
ft_half_10, f1_ft_half, accuracy_ft_half = evaluate(predictions)

# untrained
classifier_model = AutoModelForSequenceClassification.from_pretrained(base_model)
classifier = pipeline(
    "sentiment-analysis", model=classifier_model, tokenizer=tokenizer, device=0
)
predictions = data.map(lambda row: predict(row, classifier, og_label2id))
print("\nOriginal:")
og_10, f1, accuracy = evaluate(predictions)

print(
    "\ntrue:",
    data[0:9]["label"],
    "\nfine-tuned:",
    ft_10,
    "\nfine-tuned-half:",
    ft_half_10,
    "\noriginal:",
    og_10,
)
# print("\ntrue:", data[0:9]['label'], "\nfine-tuned:", ft_10, "\noriginal:", og_10)
