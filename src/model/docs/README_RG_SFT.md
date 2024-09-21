# Response Generator Fine Tuning (SFT)

## Goal

- 讓模型能根據修改版的 prompt 形式進行生成
- 模型生成的回應能符合我們指定的情緒

## Experiment

- 參考 Huggingface 的 [Trl](https://huggingface.co/docs/trl/index) 套件的文件，使用 Supervised Fine Tuning 對 LLM 進行初步訓練

### Environment

- Linux
- Nvidia RTX 3060 12G
- Python >3.11

### Steps

### 1. Choose LLM

- 在進行微調之前，先根據 baseline 的表現

### 2. Modify Prompt Template

### 3. Tune Hyperparameters

## Problem

- OOM (Out of Memory)

  - 成因：模型進行 Quantization 後，訓練仍需要使用超過當前 GPU 的記憶體 (12 G)
  - 解方：使用 [Flash Attention](https://github.com/Dao-AILab/flash-attention)
  - 由於 Transformer 架構中注意力分數計算所需的記憶體資源和時間會隨輸入序列的長度而增長，而 Flash Attention 使用線性方式來計算注意力分數，因此使用 Flash Attention 來替換原本的注意力機制能大幅減低需要的記憶體，且簡化的計算也能加速模型的推理速度。｀｀

## Result
