import torch
from tqdm.auto import tqdm

from SimilarityAnalyser import SimilarityAnalyser

test_data: list = [
    {
        "representations": [torch.clamp(torch.rand(7, dtype=torch.float32), min=-1, max=1) for _ in range(7)],
        "ideal_representation": torch.clamp(torch.rand(7, dtype=torch.float32), min=-1, max=1)
    } for _ in range(10000)
]

for threshold in tqdm(torch.range(0.5, 1, 0.01)):
    for sample in tqdm(test_data, colour="green"):
        analyser = SimilarityAnalyser(threshold)
        analyser(sample["representations"], sample["ideal_representation"])
        print(analyser.get_most_similar_representation_index())
