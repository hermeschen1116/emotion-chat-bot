# RLHF dataset (Best of N)
資料集[Shotaro30678/rlhf-RG-trl-style-v3](https://huggingface.co/datasets/Shotaro30678/rlhf-RG-trl-style-v3)

## 檔案架構
- 主程式 `bons.py`
- 參數 `args/best_of_arg.json`

## 訓練目標

- 創建用於 Direct Preference Optimization (DPO) 的 RLHF 資料集
  - 格式參照 [DPODataCollatorWithPadding](https://huggingface.co/docs/trl/main/en/dpo_trainer#expected-dataset-format) 的預設格式

## 實驗過程

透過 Best of N 來進行採樣，並且利用後續的評分機制來取得最高分以及最低分回覆，分別作為 `chosen` 以及 `rejected`。

  - **基準資料集** [hermeschen1116/daily_dialog_for_RG](https://huggingface.co/datasets/hermeschen1116/daily_dialog_for_RG)

    從基準資料集中取出 2048 項，並且進行預處理。

  - **基準模型** [hermeschen1116/response_generator_for_emotion_chat_bot](https://huggingface.co/hermeschen1116/response_generator_for_emotion_chat_bot) 
   
    利用該模型生成回覆。

  - **建立資料集**

    - **Best of N**

      透過將輸入重複成一個陣列，例如 `"how are you ?"` 變成 `["how are you ?", "how are you ?", "how are you ?"...]`，也就是說模型會從**一個輸入**產生 **N 個回覆**，相關設定如下：
      ```python
      N_BEST_OF = 6

      gen_kwargs = {
        "min_new_tokens": 5,
        "max_new_tokens": 20,
        "repetition_penalty": 1.5,
        "top_k":5,
        "top_p":1.0, 
        "temperature":2.0,
        "pad_token_id":tokenizer.pad_token_id,
        "eos_token_id":tokenizer.eos_token_id,
        "do_sample":"False"
      }
      ```
    - **評分機制**
    
      透過整合各個部分的分數來計算生成回覆的最終總分。

      - *情感分數 emotion_scores*

        透過情感分析模型 [Shotaro30678/emotion_text_classifier_on_dd_v1](https://huggingface.co/Shotaro30678/emotion_text_classifier_on_dd_v1) 來將生成出的回覆進行情感分析，並且與預期的情感比較。若情感相同則將結果的**信心分數\*10**，反之則**信心分數\*0**。

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
        透過 [madhurjindal/autonlp-Gibberish-Detector-492513457](https://huggingface.co/madhurjindal/autonlp-Gibberish-Detector-492513457) 來判斷回覆屬於 `Clean`, `Mild gibberish`, `Word salad`, `Noise` 中的哪一項。其中結果為 `Clean` **信心分數\*10** , `Mild gibberish` **信心分數\*5** ,其餘 **信心分數\-2** 使其成為負數。

## 數據