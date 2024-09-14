import os
import torch
from datasets import load_dataset, Dataset
from sklearn.metrics import f1_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from dotenv import load_dotenv
from sklearn.metrics import classification_report, confusion_matrix

load_dotenv()

new_model = "/home/user/github/etc_on_dd-1"
new_model_half = "/home/user/github/etc_on_dd-half_neutral"
base_model = "michellejieli/emotion_text_classifier"
tokenizer = AutoTokenizer.from_pretrained(base_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 7
id2label = {
    0: "neutral",
    1: "anger",
    2: "disgust",
    3: "fear",
    4: "happiness",
    5: "sadness",
    6: "surprise"
}

label2id = {v: k for k, v in id2label.items()}

og_label2id = {
    "neutral": 0,
    "anger": 1,
    "disgust": 2,
    "fear": 3,
    "joy": 4,
    "sadness": 5,
    "surprise": 6
}

def preprocessing(data):
    data = data.rename_column("utterance", "text")
    data = data.rename_column("emotion", "label")
    data = data.remove_columns("turn_type")
    return data

def shifting(data):
    df = data.to_pandas()
    df["label"] = df.groupby('dialog_id')["label"].shift(-1)
    df.dropna(inplace = True)
    df["label"]  = df["label"].astype(int)
    modified_dataset = Dataset.from_pandas(df)
    data = modified_dataset
    return data

def predict(row, classifier, label2id):
    text = row['text']
    true_label = row['label']
    predicted_result = classifier(text)[0]
    predicted_label = label2id[predicted_result["label"]]
    # print("predicted_result", predicted_result, "predicted_label", predicted_label, "true_label", true_label)

    return {"predicted_label": predicted_label, "true_label": true_label}

data_name = "benjaminbeilharz/better_daily_dialog"
data = load_dataset(data_name, split='test', num_proc=8)
data = preprocessing(data)
data = shifting(data)
data

def create_classifier_pipeline(model_name, num_labels, id2label, label2id, tokenizer):
    classifier_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id
    )
    classifier = pipeline(
        "sentiment-analysis",
        model=classifier_model,
        tokenizer=tokenizer,
        device=0
    )
    return classifier

# cm_labels = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]

def evaluate(predictions):
    true_labels = [p["true_label"] for p in predictions]
    predicted_labels = [p["predicted_label"] for p in predictions]
    print(classification_report(true_labels, predicted_labels, target_names=[id2label[i] for i in range(num_labels)], digits=4))
    first10 = predicted_labels[0:9]
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    acc = accuracy_score(true_labels, predicted_labels)
    return first10, f1, acc

# full dataset 10 epoches
classifier = create_classifier_pipeline(new_model, num_labels, id2label, label2id, tokenizer)
predictions = data.map(lambda row: predict(row, classifier, label2id))
print("Fine-tuned:")
ft_10, f1_ft, accuracy_ft = evaluate(predictions)

# half dataset 5 epoches
classifier = create_classifier_pipeline(new_model_half, num_labels, id2label, label2id, tokenizer)
predictions = data.map(lambda row: predict(row, classifier, label2id))
print("\nFine-tuned-half:")
ft_half_10, f1_ft_half, accuracy_ft_half = evaluate(predictions)

# untrained
classifier_model = AutoModelForSequenceClassification.from_pretrained(base_model)
classifier = pipeline("sentiment-analysis", model=classifier_model, tokenizer=tokenizer, device=0)
predictions = data.map(lambda row: predict(row, classifier, og_label2id))
print("\nOriginal:")
og_10, f1, accuracy = evaluate(predictions)

print("\ntrue:", data[0:9]['label'], "\nfine-tuned:", ft_10, "\nfine-tuned-half:", ft_half_10, "\noriginal:", og_10)
# print("\ntrue:", data[0:9]['label'], "\nfine-tuned:", ft_10, "\noriginal:", og_10)