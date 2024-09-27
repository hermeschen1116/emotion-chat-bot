# Sentiment Analysis

## 檔案架構

使用方式範例

```bash
uv run python sa_fine_tune.py --json_file args/sa_fine_tune_arg.json
```

- 主程式 `sa_fine_tune.py`
- 參數 `args/sa_fine_tune_arg.json`

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
    - **Downsampling**

      透過 `throw_out_partial_row_with_a_label` 將訓練資料集中 neutral 的資料移除一部分。比例可以根據 arg 中 `neutral_keep_ratio` 調整。

	  ```python
	  train_dataset = throw_out_partial_row_with_a_label(dataset["train"], run.config["neutral_keep_ratio"], 0)
	  ```


    - **Tokenization**
  
      利用 tokenizer 將 `text` 轉為 input_ids 供 Trainer 使用。

	  ```python
      train_dataset = train_dataset.map(
          lambda samples: {
            "input_ids": [tokenizer.encode(sample, padding="max_length", truncation=True) for sample in samples],
          },
          input_columns=["text"],
          batched=True,
          num_proc=16,
      )
      train_dataset.set_format("torch")
	  ```

    - **最佳化**

      - *PEFT & LoRA*

        ```python
        peft_config = LoraConfig(
          task_type="SEQ_CLS",
          lora_alpha=64,
          lora_dropout=0.2,
          r=128,
          bias="none",
          init_lora_weights=True,
          use_rslora=True,
        )
        base_model = get_peft_model(base_model, peft_config)
        ```

  - **訓練參數**

	  - *TrainingArguments*

		這裡我們選用 `paged_adamw_32bit` 作為 Optimizer，提供更好的效能。

	    ```python
        trainer_arguments = TrainingArguments(
            output_dir="./checkpoints",
            overwrite_output_dir=True,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            lr_scheduler_type="constant",
            weight_decay=0.1,
            max_grad_norm=0.3,
            num_train_epochs=5,
            warmup_ratio=0.03,
            max_steps=-1,
            logging_steps=logging_steps,
            log_level="error",
            save_steps=500,
            save_total_limit=2,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            fp16=True,
            bf16=False,
            dataloader_num_workers=12,
            optim="paged_adamw_32bit",
            group_by_length=True,
            report_to=["wandb"],
            hub_model_id="sentiment_analysis_for_emotion_chat_bot",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": True},
            auto_find_batch_size=True,
            torch_compile=False,
            include_tokens_per_second=True,
            include_num_input_tokens_seen=True,
        )
	    ```

## 數據
使用方式範例

```bash
uv run python sa_evaluate.py --json_file args/sa_evaluate_arg.json
```

- 主程式 `sa_evaluate.py`
- 參數 `args/sa_evaluate_arg.json`

- **模型表現**
  - *訓練後:*
    ```
                  precision    recall  f1-score   support

         neutral       0.90      0.88      0.89      6321
           anger       0.26      0.18      0.21       118
         disgust       0.14      0.04      0.07        47
            fear       0.00      0.00      0.00        17
       happiness       0.53      0.61      0.56      1019
         sadness       0.24      0.26      0.25       102
        surprise       0.31      0.48      0.38       116

        accuracy                           0.81      7740
       macro avg       0.34      0.35      0.34      7740
    weighted avg       0.82      0.81      0.81      7740
    ```
  - *[基準模型](https://huggingface.co/michellejieli/emotion_text_classifier):*
    ```
                  precision    recall  f1-score   support

         neutral       0.88      0.85      0.87      6321
           anger       0.21      0.35      0.26       118
         disgust       0.15      0.28      0.19        47
            fear       0.06      0.53      0.11        17
       happiness       0.60      0.37      0.46      1019
         sadness       0.13      0.38      0.20       102
        surprise       0.21      0.48      0.29       116

        accuracy                           0.77      7740
       macro avg       0.32      0.46      0.34      7740
    weighted avg       0.81      0.77      0.78      7740
    ```


## 版本迭代

- *The model did not return a loss from the inputs, only the following keys: logits. For reference, the inputs it received are input_ids,attention_mask.*


    如果 column 格式不對 `df.rename({'utterance': 'text', 'emotions': 'label'}, axis=1)` [會讓預設的讀不到，一定要是這個形式。](https://discuss.huggingface.co/t/the-model-did-not-return-a-loss-from-the-inputs-only-the-following-keys-logits-for-reference-the-inputs-it-received-are-input-values/25420/13)

- *Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your features (``label`` in this case) have excessive nesting (inputs type ``list`` where type ``int`` is expected).*
   
    label 是英文 (anger, disgust...) 而非 int

