from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import pipeline

dataset = load_dataset(
	"hermeschen1116/daily_dialog_for_RG",
	split="validation",
	keep_in_memory=True,
	num_proc=16,
	trust_remote_code=True
)

analyzer = pipeline(
    "sentiment-analysis",
    model="Shotaro30678/emotion_text_classifier_on_dd_v1",
    device='cuda'
    )

raw_analyzer = pipeline(
    "sentiment-analysis",
    model="michellejieli/emotion_text_classifier",
    device='cuda'
    )

sent_pred = []
sent_true = []
raw_sent_pred = []

for data in dataset:
    sent_true.append(data['prompt'][-1]['content']['emotion'])
    sent_pred.append(analyzer(data['prompt'][-1]['content']['dialog'])[0]['label'])
    
    raw_res = raw_analyzer(data['prompt'][-1]['content']['dialog'])[0]['label']
    if raw_res == 'joy':
        raw_sent_pred.append('happiness')
        continue
    else:
        raw_sent_pred.append(raw_res)
        
print(classification_report(sent_true, sent_pred))
print(classification_report(sent_true, raw_sent_pred))