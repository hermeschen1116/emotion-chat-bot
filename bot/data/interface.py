from typing import Any

import pytorch_lightning as pl
from datasets import load_dataset


class Interface(pl.LightningDataModule):
    def __init__(self):
        self.train_dataset: Any = None
        self.validation_dataset: Any = None
        self.test_dataset: Any = None

    def prepare_data(self) -> None:
        load_dataset("daily_dialog", download_mode="force_redownload", num_proc=8)

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset, self.validation_dataset = load_dataset("daily_dialog",
                                                                       split=["train", "validation"],
                                                                       num_proc=8)

        if stage == "test":
            self.test_dataset = load_dataset("daily_dialog", split="test", num_proc=8)

    def train_dataloader(self) -> TRAIN_DATALOADERS:

    def val_dataloader(self) -> EVAL_DATALOADERS:

    def test_dataloader(self) -> EVAL_DATALOADERS:
