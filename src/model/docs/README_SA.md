# Sentiment Analysis

## 檔案架構

使用方式範例

```bash
fine_tune_SA.py --json_file args/sa_arg.json
```

- 主程式 `fine_tune_SA.py`
- 參數 `args/sa_arg.json`

## 訓練目標

- 在 DailyDialog 資料集上微調 sentiment analysis model 以準確分析情感

## 實驗過程

利用 huggingface 的 load_dataset 來直接存取資料集，並且整理成可用的形式。例如更改 feature 名稱來對應需求。
使用 AutoTokenizer 以及 AutoModelForSequenceClassification 來產生需要的組件，並且使用 trainer.train() 進行訓練。透過 transformers 內的 [trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer) 進行微調。

  - **訓練資料集** [Shotaro30678/daily_dialog_for_SA](https://huggingface.co/datasets/Shotaro30678/daily_dialog_for_SA)

    - **預處理**
  
		原始資料集 [benjaminbeilharz/better_daily_dialog](https://huggingface.co/datasets/benjaminbeilharz/better_daily_dialog) 中包含 : `dialog_id`, `utterance`, `turn_type`, `emotion`
		我們取出 `utterance`, `emotion` 作為 `text` 以及 `label`。

  - **基準模型** [michellejieli/emotion_text_classifier](https://huggingface.co/michellejieli/emotion_text_classifier) 

  - **訓練**
    - *Downsampling*

      透過 *throw_out_partial_row_with_a_label* 將訓練資料集中 neutral 的資料移除一部分。比例可以根據 arg 中 *neutral_keep_ratio* 調整。

	  ```python
	  train_dataset = throw_out_partial_row_with_a_label(dataset["train"], run.config["neutral_keep_ratio"], 0)
	  ```


    - *Tokenization*
  
      利用 tokenizer 將 `text` 轉為 input_ids 供 Trainer 使用。

	  ```python
	  def tokenize(batch):
      	return tokenizer(batch["text"], padding="max_length", truncation=True)

	  emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
	  ```

	- **訓練參數**

	  - *TrainingArguments*

		這裡我們選用 `paged_adamw_32bit` 作為 Optimizer，提供更好的效能。

	    ```python
	    training_args = TrainingArguments(
			output_dir="./checkpoints",
			num_train_epochs=2,
			per_device_train_batch_size=batch_size,
			per_device_eval_batch_size=batch_size,
			gradient_accumulation_steps=1,
			optim="paged_adamw_32bit",
			save_steps=25,
			logging_steps=logging_steps,
			learning_rate=2e-5,
			weight_decay=0.01,
			fp16=False,
			bf16=False,
			max_grad_norm=0.3,
			max_steps=-1,
			warmup_ratio=0.03,
			group_by_length=True,
			lr_scheduler_type="constant",
			report_to=["wandb"],
			gradient_checkpointing=True,
			gradient_checkpointing_kwargs={"use_reentrant": True},
			eval_strategy="epoch",
			log_level="error"
		)
	    ```

## 版本迭代

- *The model did not return a loss from the inputs, only the following keys: logits. For reference, the inputs it received are input_ids,attention_mask.*


    如果 column 格式不對 `df.rename({'utterance': 'text', 'emotions': 'label'}, axis=1)` [會讓預設的讀不到，一定要是這個形式。](https://discuss.huggingface.co/t/the-model-did-not-return-a-loss-from-the-inputs-only-the-following-keys-logits-for-reference-the-inputs-it-received-are-input-values/25420/13)

- *Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your features (``label`` in this case) have excessive nesting (inputs type ``list`` where type ``int`` is expected).*
   
    label 是英文 (anger, disgust...) 而非 int

