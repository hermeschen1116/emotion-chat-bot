# Emotion Model

> [模型](https://huggingface.co/hermeschen1116/emotion_model_for_emotion_chat_bot)、[資料集](https://huggingface.co/datasets/hermeschen1116/emotion_transition_from_dialog)

## Goal

使模型能學習情緒的移轉，根據輸入的情緒來調整模型現在的情緒狀態

## Architectures

- **情緒狀態表徵**

  - 描述模型當前的情緒狀態

  - 一組由 7 個範圍在 [1, -1] 之間的浮點數組成的向量

  - 各個浮點數表示的是 7 種基本情緒的強度，依序是無情緒（Neutral）、憤怒（Anger）、噁心（Disgust）、恐懼（Fear）、開心（Happiness）、悲傷（Sadness）和驚訝（Surprise）

- **情緒組成**

  - 描述輸入文句的情緒組成

  - 由一組 7 個浮點數組成，所有元素總和為 1

  - 各數值同樣依序表示 7 種基本情緒的組成

## Experiment

### Environment

- Linux

- Nvidia RTX 3060 12G

- Python >3.11

### Hyperparameters Tuning

[Report](https://wandb.ai/hermeschen1116/emotion-chat-bot-ncu/reports/Emotion-Model-Hyperparamters-Tuning--Vmlldzo5NDUxOTg0)

## Result

![Baseline and SFT Comparison](images/baseline-sft-comparison.png)

|          | Accuracy           | F1 Score           |
| -------- | ------------------ | ------------------ |
| Final    | 0.7879999876022339 | 0.7950077652931213 |
| Baseline | 0.7179999947547913 | 0.7603256702423096 |
