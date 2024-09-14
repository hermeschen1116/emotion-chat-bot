# Emotion Predictor

[Shotaro30678/emotion_predcitor_for_emotion_chat_bot](https://huggingface.co/Shotaro30678/emotion_predcitor_for_emotion_chat_bot)

## 檔案架構

使用方式範例

```bash
python fine_tune_SA.py
```
- `fine_tune.py` : 主程式 
- `evaluate.py` : 和未訓練模型比較
- `sweep.py` : 利用 `wandb sweep agent` 來找出最佳 hyperparameters

## 訓練目標

- 在 DailyDialog 資料集上微調 sentiment analysis model 以準確**預測未來情感**

利用 huggingface 的 load_dataset 來直接存取資料集，並且整理成可用的形式。例如更改 feature 名稱來對應需求。
使用 AutoTokenizer 以及 AutoModelForSequenceClassification 來產生需要的組件，並且使用 trainer.train() 進行訓練。透過 transformers 內的 [trainer](https://huggingface.co/docs/transformers/main/en/main_classes/trainer) 進行微調。

  - **訓練資料集** [benjaminbeilharz/better_daily_dialog](https://huggingface.co/datasets/benjaminbeilharz/better_daily_dialog)

    - **預處理**

      - *取出需要的 column*

        該資料集中包含 : `dialog_id`, `utterance`, `turn_type`, `emotion`
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
      - *減少採樣*

		![emotion_distribution](https://github.com/user-attachments/assets/77fcd647-d524-4181-82fe-131e461a0e86)

        透過分析資料集我們發現資料集中 `neutral` 的資料量過大，可能影響訓練效果，所以我們將訓練資料集中的 neutral 資料量減半來改善效能。

        ```python
        def remove_half_train(data):
            data_set = data["train"]
            label_0_indices = [i for i, row in enumerate(data_set) if row['label'] == 0]
            num_to_remove = len(label_0_indices) // 2
            indices_to_remove = random.sample(label_0_indices, num_to_remove)
            filtered_data = data_set.filter(lambda x, i: i not in indices_to_remove, with_indices=True)
            data["train"] = filtered_data
            return data
        ```

  - **基準模型** [michellejieli/emotion_text_classifier](https://huggingface.co/michellejieli/emotion_text_classifier) 

  - **訓練**
    - *tokenization*
  
      利用 tokenizer 將 `text` 轉為 input_ids 供 Trainer 使用。

	  ```python
	  def tokenize(batch):
      	return tokenizer(batch["text"], padding="max_length", truncation=True)

	  emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
	  ```

	- **訓練參數**

    - *使用 PEFT LoRA*

      ```python
      lora_config = LoraConfig(
          lora_alpha=64,
          lora_dropout=0.2,
          r=128,
          bias="none",
          task_type="SEQ_CLS",
          use_rslora = True
      )

      peft_model = get_peft_model(model, lora_config)
      ```

    - *TrainingArguments*

		這裡我們選用 `paged_adamw_32bit` 作為 Optimizer，提供更好的效能。

	    ```python
	    training_args = TrainingArguments(
          output_dir="./checkpoints",
          num_train_epochs=5,
          load_best_model_at_end = True,
          per_device_train_batch_size=batch_size,
          per_device_eval_batch_size=batch_size,
          gradient_accumulation_steps=1,
          optim="paged_adamw_32bit",
          save_steps=500,
          save_total_limit=2,
          save_strategy = "epoch",
          logging_steps=logging_steps,
          learning_rate=0.0001,
          weight_decay=0.1,
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
          evaluation_strategy="epoch",
          log_level="error",
          overwrite_output_dir=True
      )
	    ```

## 數據
- **Fine-tuned: (10 epoches)**
  ```
                precision    recall  f1-score   support
  
       neutral     0.8159    0.9903    0.8946      5454
         anger     0.0000    0.0000    0.0000       102
       disgust     0.0000    0.0000    0.0000        41
          fear     0.0000    0.0000    0.0000        14
     happiness     0.5299    0.0669    0.1188       927
       sadness     0.0000    0.0000    0.0000        94
      surprise     0.0000    0.0000    0.0000       108
  
      accuracy                         0.8105      6740
     macro avg     0.1923    0.1510    0.1448      6740
  weighted avg     0.7331    0.8105    0.7403      6740
  ```
- **Fine-tuned: 最終決定版 (5 epoches half neutral data)**

  ```
                precision    recall  f1-score   support
  
       neutral     0.8286    0.9556    0.8876      5454
         anger     0.6000    0.0294    0.0561       102
       disgust     0.0000    0.0000    0.0000        41
          fear     0.0000    0.0000    0.0000        14
     happiness     0.4244    0.2028    0.2745       927
       sadness     0.0000    0.0000    0.0000        94
      surprise     0.0000    0.0000    0.0000       108
  
      accuracy                         0.8016      6740
     macro avg     0.2647    0.1697    0.1740      6740
  weighted avg     0.7380    0.8016    0.7568      6740
  ```
- **Baseline**

  ```
                precision    recall  f1-score   support
  
       neutral     0.8290    0.8188    0.8239      5454
         anger     0.0970    0.1569    0.1199       102
       disgust     0.0548    0.0976    0.0702        41
          fear     0.0000    0.0000    0.0000        14
     happiness     0.3987    0.2060    0.2717       927
       sadness     0.0354    0.0957    0.0517        94
      surprise     0.0079    0.0185    0.0111       108
  
      accuracy                         0.6955      6740
     macro avg     0.2033    0.1991    0.1926      6740
  weighted avg     0.7281    0.6955    0.7072      6740
  ```
- **小樣本測試結果**

  ```
  true: [6, 0, 0, 0, 0, 0, 0, 0, 0] 
  fine-tuned: [0, 0, 0, 0, 0, 0, 0, 0, 0] 
  fine-tuned-half: [0, 0, 0, 0, 0, 0, 1, 0, 4] 
  original: [0, 0, 0, 0, 0, 0, 1, 6, 1]
  ```
