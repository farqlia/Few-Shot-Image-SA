from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from fsimgsa.data.fer2013 import get_fer2013_transform, FER2013Dataset


class FER2013DataModule(pl.LightningDataModule):
    def __init__(
        self,
        transform_cfg: dict,
        data_path: str,
        batch_size: int,
        val_split: float,
        shuffle: bool,
        num_workers: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.train_ds, self.val_ds = None, None
        self.num_classes = None 
        self.classes = None
        self.hparams.dataset_cls = FER2013Dataset
        self.hparams.transform = get_fer2013_transform(transform_cfg.fer2013)

    def setup(self, stage: None | str = None):
        self.test_ds = self.hparams.dataset_cls(
            data_path=self.hparams.data_path,
            split="test",
            transform=self.hparams.transform,
        )
        dataset = self.hparams.dataset_cls(
            data_path=self.hparams.data_path,
            split="train",
            transform=self.hparams.transform,
        )
        self.num_classes = dataset.num_classes
        self.classes = dataset.classes
        train_split = 1 - self.hparams.val_split
        self.train_ds, self.val_ds = random_split(
            dataset, [train_split, self.hparams.val_split]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            persistent_workers=True,
        )
