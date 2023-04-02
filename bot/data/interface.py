import pytorch_lightning as pl


class Interface(pl.LightningDataModule):
    def __init__(self):

    def setup(self, stage: str) -> None:

    def train_dataloader(self) -> TRAIN_DATALOADERS:

    def val_dataloader(self) -> EVAL_DATALOADERS:

    def test_dataloader(self) -> EVAL_DATALOADERS:
