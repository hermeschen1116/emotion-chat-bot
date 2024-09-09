# Response Generator further fintuning (RLHF)

## 檔案架構

使用方式範例

```bash
fine_tune_dpo.py --json_file args/dpo_arg.json
```

### DPO 
- 主程式 `fine_tune_dpo.py`
- 參數 `args/dpo_arg.json`

### Evaluation
- 主程式 `evaluate.py`
- 參數 `args/evaluate_arg.json`

## 訓練目標

- 在自訂的資料集上利用 RLHF 來讓模型對齊需求並且改善效能。

## 實驗過程

透過 TRL 中的 [DPO trainer](https://huggingface.co/docs/trl/main/en/dpo_trainer) 來進行訓練。

  - **訓練資料集** [Shotaro30678/rlhf-RG-trl-style-v3](https://huggingface.co/datasets/Shotaro30678/rlhf-RG-trl-style-v3)

    該資料集中包含：
    - *prompt*
    - *chosen*
    - *rejected*
    - *chosen_score*
    - *rejected_score*

    其中 `chosen_score` 以及 `rejected_score` 僅作為資料集評估以及參考用，並不會作為訓練用的輸入。

  - **基準模型** [hermeschen1116/response_generator_for_emotion_chat_bot](https://huggingface.co/hermeschen1116/response_generator_for_emotion_chat_bot) 

  - **訓練**

    - **最佳化**
      - *unsloth*

        ```python
        from unsloth import FastLanguageModel, PatchDPOTrainer
        PatchDPOTrainer()

        base_model, tokenizer = FastLanguageModel.from_pretrained(
            wandb.config["base_model"],
            attn_implementation="flash_attention_2",
            pretraining_tp=1,
            load_in_4bit=True,
            use_cache=False,
            device_map="auto",
            use_gradient_checkpointing="unsloth",
            low_cpu_mem_usage=True,
        )
        ```

        [<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)

      - Reference model considerations with PEFT

        除了 unsloth 之外，我們也透過[載入兩次 adapter](https://huggingface.co/docs/trl/main/en/dpo_trainer#using-option-3---load-the-adapter-twice)的方式來進一步降低 reference model 的記憶體。

        ```python
        model = PeftModel.from_pretrained(
            base_model,
            wandb.config["adapter"],
            is_trainable=True,
            adapter_name="traingg",
        ) 

        model.load_adapter(
            wandb.config["adapter"],
            adapter_name="reference"
        )

        training_args = DPOConfig(
            model_adapter_name="traingg",
            ref_adapter_name="reference",
            ...
        )
        ```
    - **參數設定**

      DPO 訓練參數

      ```python
      training_args = DPOConfig(
          output_dir="./checkpoints",
          beta=0.1,
          model_adapter_name="traingg",
          ref_adapter_name="reference",
          remove_unused_columns=False,
          num_train_epochs=3,
          gradient_checkpointing=True
      )
      ```
      其餘維持預設。

  - **評估**
  
    我們利用 [hermeschen1116/daily_dialog_for_RG](https://huggingface.co/datasets/hermeschen1116/daily_dialog_for_RG) 的 `test` 來進行模型結果評估

    ```python
    dataset = load_dataset(
      "hermeschen1116/daily_dialog_for_RG",
      split="test",
      num_proc=16,
      trust_remote_code=True
    )  
    ```
    - **參數設定**
    
      根據[官方文件](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig)以及交叉測試，我們發現 `multinomial sampling` 的結果最爲穩定平衡。

      ```python
      generation_config = GenerationConfig(
          max_new_tokens=150,
          min_new_tokens=5,
          repetition_penalty=1.1,
          top_k=3,
          top_p=0.9,
          pad_token_id=tokenizer.pad_token_id,
          eos_token_id=tokenizer.eos_token_id,
          temperature=1.0,
          do_sample=True,
          num_beams=1
      )
      ```

    - **評估指標**
      WIP

## 數據

- **生成時長**

  在輸入為 2048 行的情況下，我們在單張 `RTX 3090` 共花費了 `12h 4m 21s` 生成。

- **分數分佈**
  
  這邊的分數指 `[chosen_score] - [rejected_score]`

  ![alt text](image.png)

  |指標     |分數       |
  |--------|----------|
  |Median  |**17.547**|
  |Mean    |**16.144**|
  |SD      |**5.815**|

## 版本迭代

- v1:建立基本流程，但是分數未設定限制導致差距小且資料集分數整體偏低。
- v2:改善分數計算，但時常出現 `chosen` 中有 `cut-off` 的回覆。
- v3:透過強制篩選 `chosen` 來精準控制結果，並且捨棄 `cut-off` 的回覆。