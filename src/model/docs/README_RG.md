# Response Generator

> [模型](https://huggingface.co/hermeschen1116/response_generator_for_emotion_chat_bot)、[資料集](https://huggingface.co/datasets/hermeschen1116/daily_dialog_for_RG)

## Tech Stack

- PEFT (LoRA)
- TRL (SFT, DPO)
- BitsAndBytes
- Unsloth

## Experiement

### Goals

- 生成符合指定情緒的回應
- 生成如同日常對話的回應

### Methods

- **Base Model**: [Llama 2 7B](https://huggingface.co/unsloth/llama-2-7b-bnb-4bit)
- **Dataset**: [Daily Dialog](https://huggingface.co/datasets/li2017dailydialog/daily_dialog)

#### Steps

- [**Fine Tune**](README_RG_SFT.md)

  - 使用 Supervised Fine Tuning 在 Daily Dialog 進行初步的訓練
  - 訓練成效
    ![Baseline and SFT Comparison](images/baseline-sft-comparison.png)
  - 詳細可參考[**此頁面**](README_RG_SFT.md)

- [**Further Fine Tune**](README_RG_DPO.md)

- 使用 Direct Preference Optimization 在合成資料集上進行初步的訓練
- 詳細可參考[**此頁面**](README_RG_DPO.md)
