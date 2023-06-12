from typing import Any

import fasttext
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader


class Interface(pl.LightningDataModule):
    def __init__(self, path_to_embedding_model: str):
        super().__init__()
        self.__train_dataset: Any = None
        self.__validation_dataset: Any = None
        self.__test_dataset: Any = None
        self.__embedding_model: Any = Preprocess(path_to_embedding_model)

    def prepare_data(self) -> None:
        # noinspection PyTypeChecker
        load_dataset("daily_dialog",
                     download_mode="force_redownload",
                     num_proc=8)

    def setup(self, stage: str) -> None:
        dataset: Any = load_dataset("daily_dialog", num_proc=8)
        dataset = dataset.map(self.__embedding_model.get_embedding)
        dataset.set_format(type='torch', columns=['dialog'])
        dataset = dataset.shuffle(seed=37710)
        if stage == "fit":
            self.__train_dataset, self.__validation_dataset = dataset['train'], dataset['validation']

        if stage == "test":
            self.__test_dataset = dataset['test']

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.__train_dataset, batch_size=1, num_workers=8)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.__validation_dataset, batch_size=1, num_workers=8)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.__test_dataset, batch_size=1, num_workers=8)


class Preprocess:
    def __init__(self, path_to_embedding_model: str):
        self.__embedding_model: Any = fasttext.load_model(path_to_embedding_model)

    def get_embedding(self, samples: Any) -> dict:
        processed_samples: Any = samples
        if type(processed_samples) is list:
            for sample in processed_samples:
                sample['dialog'] = [

                    [self.__embedding_model.get_word_vector(word) for word in sentence.strip().split(' ')]
                    for sentence in sample['dialog']]
        else:
            processed_samples['dialog'] = [
                [self.__embedding_model.get_word_vector(word) for word in sentence.strip().split(' ')]
                for sentence in processed_samples['dialog']]
        return processed_samples
