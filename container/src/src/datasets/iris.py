import os
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split


class IRISDataset(LightningDataModule):

    name = "IRIS"

    def __init__(
        self,
        data_dir: str,
        val_split: float = 0.3,
        num_workers: int = 16,
        P=None,
        seed: int = 42,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._data_dir = data_dir
        self._val_split = val_split
        self.num_workers = num_workers
        self._P = P
        self._seed = seed
        assert self._P is not None

    @property
    def num_classes(self):
        return 1

    def _labelize(self, labels):
        self.unique_labels np.unique(labels)
        for idx, u in enumerate(self.unique_labels):
            labels[labels == u] = idx

    def prepare_data(self):
        input_files = [
            os.path.join(self._P.TRAINING_PATH, filename)
            for filename in os.listdir(self._P.TRAINING_PATH)
        ]
        if len(input_files) == 0:
            raise ValueError(
                (
                    "There are no files in {}.\n"
                    + "This usually indicates that the channel ({}) was incorrectly specified,\n"
                    + "the data specification in S3 was incorrectly specified or the role specified\n"
                    + "does not have permission to access the data."
                ).format(training_path, channel_name)
            )

        raw_data = [pd.read_csv(file, header=None) for file in input_files]
        raw_data = torch.from_numpy(pd.concat(raw_data).values.astype(np.float))
        self._labelize(raw_data[:, 0])
        raw_data_length = len(raw_data)

        train_size = int(raw_data_length * (1 - self._val_split))
        val_size = raw_data_length - train_size

        self.dataset_train, self.dataset_val = random_split(
            raw_data,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self._seed),
        )

    def train_dataloader(self, batch_size=32, transforms=None):
        loader = DataLoader(
            self.dataset_train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self, batch_size=32, transforms=None):

        loader = DataLoader(
            self.dataset_val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader
