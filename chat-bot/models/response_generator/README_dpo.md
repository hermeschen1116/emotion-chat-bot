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
      - *情感分析 Sentiment analysis*

        透過自訂的情感分析模型 [Shotaro30678/emotion_text_classifier_on_dd_v1](https://huggingface.co/Shotaro30678/emotion_text_classifier_on_dd_v1) 分析生成的情感是否與目標相同。

      - *胡言亂語等級 Gibberish level*

        利用 [madhurjindal/autonlp-Gibberish-Detector-492513457](https://huggingface.co/madhurjindal/autonlp-Gibberish-Detector-492513457) 來分析回覆的胡言亂語等級，並且進行統計。

      - *不完整回覆數量 Incomplete amount*

        若回覆結尾爲 `!`, `?`, `.` 則標記爲**完整 (Complete)**，反之則爲**不完整 (Incomplete)**。

## 數據

- **生成時長**

  在 **1000 筆**的輸入下，在單張 `RTX 3090` 上大約需要 `5m` 來執行。

- **模型表現**

  - *情感分析分數 Sentiment Score:*

    | **Metric**   | **DPO Trained** | **Baseline** |
    |--------------|:----------------------:|:--------------------------:|
    | **Accuracy** | 0.851                 | 0.788                     |
    | **F1-score** | 0.8564                | 0.7975                    |

  - *胡言亂語分佈 Gibberish Distribution:*

    | **Category**        | **DPO Trained** | **Baseline** |
    |---------------------|:----------------------:|:--------------------------:|
    | **Clean**           | 882                   | 898                       |
    | **Mild Gibberish**  | 94                    | 58                        |
    | **Word Salad**      | 21                    | 33                        |
    | **Noise**           | 3                     | 11                        |

  - *不完整回覆 Cut-Off Output:*

    | **Output Type**     | **DPO Trained** | **Baseline** |
    |---------------------|:----------------------:|:--------------------------:|
    | **Complete Output** | 985                   | 975                       |
    | **Incomplete Output** | 15                  | 25                        |

## 版本迭代

- PPO: 用盡各種方法仍然會 `OOM` (單張 `RTX3090 24G`)
- RLOO: 用盡各種方法仍然會 `OOM` (單張 `RTX3090 24G`)
- DPO: 成功訓練。
- evaluate.py: 改進生成策略，新增評估方式。

## 結論
- 即便我們透過 DPO 成功訓練，但 DPO 需要建立 `preference dataset` 的模式會相當**耗時且難以保證資料集的品質**。同時 DPO 因其離線架構 (Offline-RLHF) 可能使結果在**某些情況下不如 PPO (Online-RLHF)**。

  > 附註： [Offline vs. Online Reinforcement Learning](https://huggingface.co/learn/deep-rl-course/unitbonus3/offline-online), [Online DPO Trainer](https://huggingface.co/docs/trl/main/en/online_dpo_trainer#online-dpo-trainer)

- 理論上 RLOO 應該是**相對更佳**的解決方案。RLOO 為 Online-RLHF，相比 Offline-RLHF 可以即時收集回饋並且改進模型參數。RLOO 比起 PPO 有更好的性能且需要更少的記憶體，同時也無需建立資料集。但即使如此，此方案對於我們現有的硬體來說仍然需求過高。