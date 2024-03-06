# 專題
中央大學專題

### how-to
執行 `run_SA.ipynb`

### 專題作業時程
#### Sentiment Analysis （3月初）
* Test prompt and optimize 
#### Response Generator （3月底）
* Test prompt and optimize
#### Candidate Generator （4月中）
* Test prompt
* Test different length of history
* Optimize 
* (option) use Trl to strengthen divergence 
#### Similarity Analysis （3月底）
* Find right math formula
* Experiment 
#### Emotion Model （4月底）
* Understand different types of attention mechanism 
* Build model with attention mechanism 
* Optimize
#### Full Model（5月中）
* Optimize and Improve 
* Application

### To-Do
- [X] mapping output to dailydialog style
- [X] test performance
- [ ] ...
---
### issues
1. **The model did not return a loss from the inputs, only the following keys: logits. For reference, the inputs it received are input_ids,attention_mask.**


    如果沒有 `df.rename({'utterance': 'text', 'emotions': 'label'}, axis=1)` [好像會讓他讀不到，一定要是這個形式。](https://discuss.huggingface.co/t/the-model-did-not-return-a-loss-from-the-inputs-only-the-following-keys-logits-for-reference-the-inputs-it-received-are-input-values/25420/13)

2. **Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your features (``label`` in this case) have excessive nesting (inputs type ``list`` where type ``int`` is expected).**
   
    這就忘了我的 label 是英文(anger, disgust...)