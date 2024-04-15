import os

import wandb
from dotenv import load_dotenv
import torch
from tqdm.auto import tqdm
from SimilarityAnalyser import SimilarityAnalyser

# prevent env load failed
load_dotenv(encoding="utf-8")
wandb.login(key=os.environ.get("WANDB_API_KEY", ""), relogin=True)

test_data: list = [
    {
        "representations": [torch.clamp(torch.rand(7, dtype=torch.float32), min=-1, max=1) for _ in range(7)],
        "ideal_representation": torch.clamp(torch.rand(7, dtype=torch.float32), min=-1, max=1)
    } for _ in range(10000)
]

for threshold in tqdm(torch.range(0, 1, 0.01)):
    wandb.init(project="emotion-chat-bot-ncu",
               group="Similarity Analysis",
               job_type="test",
               notes="test the best threshold and if there's collision in similarity",
               config={
                   "threshold": threshold,
                   "num_samples": 10000
               })
    for sample in tqdm(test_data, colour="green"):
        analyser = SimilarityAnalyser(threshold)
        analyser(sample["representations"], sample["ideal_representation"])

        try:
            analyser.get_most_similar_representation_index()
        except Exception as e:
            wandb.log({
                "problem": "collision"
            })

    wandb.finish()
