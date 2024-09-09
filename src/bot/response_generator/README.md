# Response Generator further fintuning (RLHF)

## 檔案架構
- 主程式 `fine_tune_dpo.py`
- 參數 `args/dpo_arg.json`

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

      除了 unsloth 之外，我們也透過載入兩次 adapter 的方式來進一步降低 reference model 的記憶體。

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
          output_dir="./checkpoints",
          beta=0.1,
          model_adapter_name="traingg",
          ref_adapter_name="reference",
          remove_unused_columns=False,
          num_train_epochs=3,
          gradient_checkpointing=True
      )
      ```
      WIP


    - **評分機制**
    
      透過整合各個部分的分數來計算生成回覆的最終總分。

      - *情感分數 emotion_scores*

        透過情感分析模型 [Shotaro30678/emotion_text_classifier_on_dd_v1](https://huggingface.co/Shotaro30678/emotion_text_classifier_on_dd_v1) 來將生成出的回覆進行情感分析，並且與預期的情感比較。
        - 情感相同則將結果的**信心分數\*10**
        - 情感不同則將結果的**信心分數\*0**

      - *長度分數 length_scores*
  
        透過級距的方式評分，根據回應長度與設定的最小和最大期望值之間的差距調整分數。

        ```python
        def length_reward(response_length: int) -> float:
            difference_ratio_min = (response_length - 5) / 5
            difference_ratio_max = (response_length - 20) / 20

            if abs(difference_ratio_min) < 1:
                return difference_ratio_min * 0.0001
            elif abs(difference_ratio_min) > 1 > abs(difference_ratio_max):
                return abs(difference_ratio_min + difference_ratio_max) * 10
            else:
                return difference_ratio_max * 0.9
        ```
        - `difference_ratio_min` 的絕對值小於 1 時，表示回應長度接近最小期望值 5，給予極小的獎勵分數。

        - `difference_ratio_min` 的絕對值大於 1 且 `difference_ratio_max` 的絕對值小於 1 時，表示回應長度偏離最小期望值 5 較多，但距離最大期望值 20 不遠，給予較大的獎勵分數。

        - 其他情況下，給予 `difference_ratio_max` 乘以 0.9 的獎勵分數。

      - *胡言亂語分數 gibberish_scores*
        透過 [madhurjindal/autonlp-Gibberish-Detector-492513457](https://huggingface.co/madhurjindal/autonlp-Gibberish-Detector-492513457) 來判斷回覆屬於 `Clean`, `Mild gibberish`, `Word salad`, `Noise` 中的哪一項。
        - 結果為 `Clean` **信心分數\*10**
        - `Mild gibberish` **信心分數\*5**
        - 其餘 **信心分數\-2**
        - 
      - *總分 reward*

        計算總分時，透過加入**權重（Weight）**以及**誤差（Bias）**的方式來平衡不同參數的重要性。
        ```python
        return [reward_weight.dot(tensor(reward_score, dtype=torch.float)) + reward_bias
                for reward_score in zip(emotion_scores, length_scores, gibberish_scores)]
        ```
        參數設定
        ```python
        "reward_weights": [0.4, 0.25, 0.35]
        "reward_bias": 0.001
        ```

    - **失敗機制**

      除了上述的評分機制，我們也採用了失敗機制來篩選不合需求的回覆並且重新生成。

      - *分數全距*

        計算出生成的回覆陣列中分數最大和最小的差，並且判斷是否**大於目標值**，以及分數最高的回覆分數是否**大於8**。不滿足則重新生成。

        ```python
        if score_range < target_score_range or max(score_tmp) < 8 :
        ```

        該步驟的目的在於篩選掉 `chosen` 和 `rejected` 差異太小的回覆組合，並且篩選掉回覆不夠完整的 `chosen` 。
      
      - *chosen 篩選*
      
        分析當前組合中的 `chosen` 是否滿足以下條件：

        - 情感是否與**目標情緒相同**
        - 胡言亂語程度**是否為 `Clean`**
        - 胡言亂語**信心分數是否 >= 0.8**
        - 結尾是否為 **"!", ".", "?"**
      
        ```python
        if (chosen_sentiment['label'] != tmp['label'] 
            or chosen_gibberish['label'] != "clean" 
            or chosen_gibberish['score'] < 0.8
            or chosen.strip(" ")[-1:] not in ["!", ".", "?"]
            ):    
        ```
        不滿足其中任何一項條件則重新生成。

      - *生成計數器*
  
        為了避免生成回覆的時間過長或是回覆一直無法滿足條件的情況，我們針對重新生成設定了**次數上限 30 次**。
        ```python
        if fail_counter <= 30:
            print("\nRegenerating...")
            continue
        else:
            fail_index.append(i)
            break
        ```
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