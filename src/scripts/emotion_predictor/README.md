# Emotion Predictor

[Shotaro30678/emotion_predcitor_for_emotion_chat_bot](https://huggingface.co/Shotaro30678/emotion_predcitor_for_emotion_chat_bot)

## 檔案架構

使用方式範例

```bash
uv run python ep_fine_tune.py --json_file args/ep_fine_tune_arg.json
```

- 主程式 `ep_fine_tune.py`
- 參數 `args/ep_fine_tune_arg.json`

## 訓練目標

- 在 DailyDialog 資料集上微調 sentiment analysis model 以準確**預測未來情感**

## 實驗過程

  利用 huggingface 的 load_dataset 來直接存取資料集。使用 AutoTokenizer 以及 AutoModelForSequenceClassification 來產生需要的組件，並且使用 trainer.train() 進行訓練。透過 transformers 內的 [trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer) 進行微調。

  - **訓練資料集** [Shotaro30678/daily_dialog_for_EP](https://huggingface.co/datasets/Shotaro30678/daily_dialog_for_EP)

    - **預處理**

      - *取出需要的 column*

        原始資料集 [benjaminbeilharz/better_daily_dialog](https://huggingface.co/datasets/benjaminbeilharz/better_daily_dialog) 中包含 : `dialog_id`, `utterance`, `turn_type`, `emotion`
        我們取出 `utterance`, `emotion` 作為 `text` 以及 `label`。

        ```python
        def preprocessing(data):
            data = data.rename_column("utterance", "text")
            data = data.rename_column("emotion", "label")
            data = data.remove_columns("turn_type")
            return data
        ```

      - *平移 label*

        ```python
        def shift_labels(dataset):
            df = dataset.to_pandas()
            df["label"] = df.groupby('dialog_id')["label"].shift(-1)
            df.dropna(inplace = True)
            df["label"]  = df["label"].astype(int)
            dataset = Dataset.from_pandas(df)
            dataset = dataset.remove_columns("dialog_id")
            return dataset
        ```

        因為要預測的目標是**未來的情緒**，也就是透過**當前對話的語句預測下一輪對話的情緒**。

        - 按對話 ID 分組，對 label 欄位進行移位操作

          ```python
          df["label"] = df.groupby('dialog_id')["label"].shift(-1)
          ```

          針對同一個 dialog_id 分組，將 label 欄位的值向上移一位（即：將當前對話的 label 替換為下一句對話的 label）。

        - 移除空值
    
          移位操作後，因為沒有後續的對話，最後一行 label 會變為 NaN，需要將這些行移除。
          ```python
          df.dropna(inplace=True)
          ```
          
      透過上述處理後的資料集即為訓練資料集 [Shotaro30678/daily_dialog_for_EP](https://huggingface.co/datasets/Shotaro30678/daily_dialog_for_EP)

  - **基準模型** [j-hartmann/emotion-english-distilroberta-base](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)

    考慮到訓練資料的差異，我們選用與 sentiment analysis 不同的模型來避免模型學習成效不佳的問題。

  - **訓練**
    - **Downsampling**
      
      透過分析資料集我們發現資料集中 `neutral` 的資料量過大，可能影響訓練效果，所以我們將訓練資料集中的 neutral 資料量減半來改善效能。

		![emotion_distribution](https://github.com/user-attachments/assets/77fcd647-d524-4181-82fe-131e461a0e86)


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
          r=32,
          bias="none",
          init_lora_weights=True,
          use_rslora=True,
        )
        base_model = get_peft_model(base_model, peft_config)

  - **訓練參數**

    - *class_weights*
  
      一開始訓練成果不彰，考慮到資料分佈不均的問題，我們透過計算出各個 class 權重的方式來改善訓練效果

      ```python
      y = train_dataset["label"].tolist()
      class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
      ```
    - *Focal Loss*

      透過 Focal Loss 我們成功讓模型對少數 class 的關注度提升，其中我們的設定為 `alpha = class_weight` , `gamma = 8`

      ```python
      loss_fct = FocalLoss(alpha=class_weights, gamma=run.config["focal_gamma"])
      ```

    - *TrainingArguments*

		這裡我們選用 `paged_adamw_32bit` 作為 Optimizer，提供更好的效能。

	    ```python
        trainer_arguments = TrainingArguments(
            output_dir="./checkpoints",
            overwrite_output_dir=True,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=0.0007627,
            lr_scheduler_type="cosine",
            weight_decay=0.3052,
            max_grad_norm=0.6186,
            num_train_epochs=5,
            warmup_ratio=0.06915,
            max_steps=-1,
            logging_steps=logging_steps,
            log_level="error",
            save_steps=500,
            save_total_limit=2,
            save_strategy="epoch",
            eval_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="F1-all-class",
            greater_is_better=True,
            fp16=True,
            bf16=False,
            dataloader_num_workers=12,
            optim="paged_adamw_32bit",
            group_by_length=True,
            report_to=["wandb"],
            hub_model_id=run.config["fine_tuned_model"],
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": True},
            torch_compile=False,
            include_tokens_per_second=True,
            include_num_input_tokens_seen=True,
        )
	    ```

    - *CustomTrainer*

      為了使用自定義的 `FocalLoss`，我們設定了 `CustomTrainer` 用於訓練。

      ```python
      class CustomTrainer(Trainer):
          def compute_loss(self, model, inputs, return_outputs=False):
              labels = inputs.get("labels")
              outputs = model(**inputs)
              logits = outputs.get("logits")
              loss = loss_fct(logits, labels)
              return (loss, outputs) if return_outputs else loss
      ````

## 數據
使用方式範例

```bash
uv run python ep_evaluate.py --json_file args/ep_evaluate_arg.json
```

- 主程式 `ep_evaluate.py`
- 參數 `args/ep_evaluate_arg.json`

- **模型表現**

  - *focal loss + class weight :*
    ```
                  precision    recall  f1-score   support

         neutral       0.89      0.24      0.38      5454
           anger       0.06      0.17      0.09       102
         disgust       0.02      0.20      0.04        41
            fear       0.01      0.71      0.01        14
       happiness       0.23      0.53      0.32       927
         sadness       0.03      0.12      0.05        94
        surprise       0.04      0.18      0.06       108

        accuracy                           0.28      6740
       macro avg       0.18      0.31      0.14      6740
    weighted avg       0.75      0.28      0.35      6740

              Accuracy 0.27819
     Balanced_Accuracy 0.30592
              F1-score 0.35445
    ```

  - *預設 loss :*

    ```
                  precision    recall  f1-score   support

         neutral       0.84      0.91      0.87      5454
           anger       0.17      0.09      0.12       102
         disgust       0.00      0.00      0.00        41
            fear       0.00      0.00      0.00        14
       happiness       0.36      0.30      0.33       927
         sadness       0.00      0.00      0.00        94
        surprise       0.00      0.00      0.00       108

        accuracy                           0.78      6740
       macro avg       0.20      0.19      0.19      6740
    weighted avg       0.73      0.78      0.75      6740

              Accuracy 0.77982
     Balanced_Accuracy 0.18522
              F1-score 0.7538
    ```
  - *[基準模型](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)*

    ```
                  precision    recall  f1-score   support

         neutral       0.82      0.52      0.64      5454
           anger       0.06      0.11      0.08       102
         disgust       0.03      0.10      0.04        41
            fear       0.00      0.29      0.00        14
       happiness       0.34      0.25      0.29       927
         sadness       0.04      0.15      0.07        94
        surprise       0.02      0.05      0.03       108

        accuracy                           0.46      6740
       macro avg       0.19      0.21      0.16      6740
    weighted avg       0.71      0.46      0.56      6740

              Accuracy 0.4638
     Balanced_Accuracy 0.20912
              F1-score 0.55973
    ```

## Hyperparameter Tuning

> [Report](https://wandb.ai/yangyx30678/emotion-chat-bot-ncu-ep-sweep/reports/Emotion-Predictor-Hyperparamters-Tuning--Vmlldzo5NTMxNjUy?accessToken=cml3ma0vvrcqlkkvlni6avpq8ymewx8fck7anmi1k66twosk8j1jsnqid86hwtmt)

- 使用 WanDB Sweep 來找到最佳化的超參數

  使用方式範例

  ```bash
  uv run python ep_sweep.py --json_file args/ep_sweep_arg.json
  ```

  - 主程式 `ep_sweep.py`
  - 參數 `args/ep_sweep_arg.json`

- 參數釋義
  - *F1-all-class*

    將 `multiclass_f1_score` 設定為 `average=None` 計算出個別 class 的分數，結果即為 `f1_per_class`。

    ```python
    f1_per_class = multiclass_f1_score(
        sentiment_true,
        sentiment_pred,
        num_classes=num_emotion_labels,
        average=None,
    ).to("cuda")
    ```

    我們將 `f1_per_class` 與 `class_weights` 相乘作為 `F1-all-class`。

    ```python
    weighted_f1_per_class = f1_per_class * class_weights
    weighted_f1_all_class = weighted_f1_per_class.mean()
    ```
  - *Classes-with-value*

    統計出分數不為零的 class 數量，判斷模型對少數 class 的關注度。

    ```python
    non_zero_count = (weighted_f1_per_class != 0).sum()
    ```
  - *Balanced_Accuracy*

    [balanced_accuracy_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.balanced_accuracy_score.html)

    Sweep 的尋找目標。相比於 `Accuracy`，`Balanced_Accuracy` 可以更好地評估不平衡資料集。定義為**每個 class recall 的平均**。



- 調整參數
  ```json
  "sweep_config" : {
  	"method": "bayes",
  	"name": "sweep",
  	"metric": {"goal": "maximize", "name": "Balanced_Accuracy"},
  	"parameters": {
  		"batch_size": {"values": [8, 32, 64]},
  		"num_train_epochs": {"values": [3, 5, 8]},
  		"learning_rate": {"max": 0.05, "min": 0.0001},
  		"lr_scheduler_type": {"values": ["constant", "cosine"]},
  		"lora_alpha": {"values": [16, 32, 64]},
  		"lora_dropout": {"values": [0.1, 0.2, 0.3]},
  		"lora_rank": {"values": [16, 32, 64]},
  		"init_lora_weights": {"values": [true, false]},
  		"use_rslora": {"values": [true, false]},
  		"focal_gamma": {"values": [1, 3, 5, 8]},
  		"weight_decay": {"max": 0.5, "min": 0.0},
  		"warmup_ratio": {"max": 0.1, "min": 0.0},
  		"max_steps": {"value": -1},
  		"max_grad_norm": {"max": 1.0, "min": 0.1}
  	}
  }
  ```
## 版本迭代

- 直接訓練: 沿用 sentiment analysis 的方法訓練。效果不佳且對少數 class 預測效果更糟。
- class_weight: 引入 class_weight，效果有所提升。
- class_weight + focal loss: class_weight 配合 focal losss，效果是目前最好的。
- oversampling: 利用 SMOTE, ADASYN... 進行重採樣，效果不增反減。