# Emotion Model

> [模型](https://huggingface.co/hermeschen1116/emotion_model_for_emotion_chat_bot)、[資料集](https://huggingface.co/datasets/hermeschen1116/emotion_transition_from_dialog)

## Goal

使模型能學習情緒的移轉，根據輸入的情緒來調整模型現在的情緒狀態

## Architectures

```python
EmotionModel(
  (_EmotionModel__attention): DotProductAttention()
  (_EmotionModel__dropout): Dropout(p=0.5321222767142184, inplace=False)
  (_EmotionModel__weight_D): Linear(in_features=7, out_features=7, bias=False)
)
```

- **情緒狀態表徵**

  - 描述模型當前的情緒狀態

  - 一組由 7 個範圍在 [1, -1] 之間的浮點數組成的向量

  - 各個浮點數表示的是 7 種基本情緒的強度，依序是無情緒（Neutral）、憤怒（Anger）、噁心（Disgust）、恐懼（Fear）、開心（Happiness）、悲傷（Sadness）和驚訝（Surprise）

  - 數值最大的元素代表模型當前的主要情緒

- **情緒組成**

  - 描述輸入文句的情緒組成

  - 由一組 7 個浮點數組成，所有元素總和為 1

  - 各數值同樣依序表示 7 種基本情緒的組成

### 架構設計思路

```python
def forward(self, representation: Tensor, input_emotion: Tensor) -> Tensor:
		representation = representation.to(dtype=self.dtype)
		input_emotion = input_emotion.to(dtype=self.dtype)

		raw_attention: Tensor = self.__attention.forward(input_emotion, representation.squeeze().diag())

		attention_score: Tensor = raw_attention.softmax(-1).squeeze().diag()

		difference: Tensor = torch.clamp(torch.diagonal(self.__weight_D((attention_score**3))), -1, 1)

		return representation + difference
```

- 分別計算情緒狀態表徵中每個元素跟輸入文句的情緒組成之間的關係（此處為注意力分數），作為調整情緒狀態的依據

- 將計算好的注意力分數三次方後通過一層 LinearLayer，結果與當前的情緒狀態表徵相加得到新的情緒狀態表徵

## Experiment

### Environment

- Linux

- Nvidia RTX 3060 12G

- Python >3.11

### 設計資料集

- 由於找不到符合要求的資料集，因此在 Daily Dialog 的基礎上使用合成的方式製作

- 將 Daily Dialog 中的多回合對話設定為由模型發起的對話，因此使用第一句的情緒來隨機生成一組情緒表徵，情緒對應的強度數值最高

- 使用我們自行微調的 Sentiment Analysis 模型來分析使用者的對話中的情緒，並將各個情緒的可能性組合成一向量後經過 Softmax 使其總和為 1

- 以此類推能得到多組多回合對話的情緒組成

### 訓練方式

```python
def representation_evolute(
	model, bot_emotion_representations: List[Tensor], user_emotion_compositions: List[Tensor]
) -> List[Tensor]:
	evolute_representations: list = bot_emotion_representations
	for composition in user_emotion_compositions:
		new_representation: Tensor = model.forward(evolute_representations[-1], composition)
		evolute_representations.append(new_representation)

	return evolute_representations
```

- 模型輸入為模型當前的情緒狀態和輸入文句的情緒組成，得到新的情緒狀態後再和下一句的情緒組成作為新的輸入，反覆進行

- 對比生成的情緒表徵中強度最大值的情緒和實際對話中的情緒來作為梯度下降的依據

### Hyperparameter Tuning

> [Report](https://wandb.ai/hermeschen1116/emotion-chat-bot-ncu/reports/Emotion-Model-Hyperparamters-Tuning--Vmlldzo5NDUxOTg0)

- 使用 WanDB Sweep 來找到最佳化的超參數

- 調整參數

  - 注意力機制：Dot Product／Scaled Dot Product／Additive／Dual Linear

  - Dropout

  - Linear Layer 的 Bias

  - 學習率跟優化器

  - 訓練次數
