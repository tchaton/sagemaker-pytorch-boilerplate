import os
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from src.paths import Paths


class BaseSagemakerDataset(LightningDataModule):

    name = "BaseSagemaker"

    def __init__(
        self, val_split: float = 0.3, P: Paths = None, *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._P = P
        self._val_split = val_split
        assert self._P is not None

    @property
    def num_features(self):
        return NotImplementedError

    @property
    def num_classes(self):
        return NotImplementedError

    def _labelize(self, raw_data):
        raise NotImplementedError

    def prepare_splitted_data(self, path_train, path_val):
        train_raw_data = pd.read_csv(path_train, header=None).values

        val_raw_data = pd.read_csv(path_val, header=None).values

        self._labelize(train_raw_data)
        self._labelize(val_raw_data)
        self.dataset_train = torch.from_numpy(train_raw_data.astype(np.float)).float()
        self.dataset_val = torch.from_numpy(val_raw_data.astype(np.float)).float()

    def _prepare_no_splitted_data(self):
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
        raw_data = pd.concat(raw_data).values
        self._labelize(raw_data)
        raw_data = torch.from_numpy(raw_data.astype(np.float)).float()
        raw_data_length = len(raw_data)

        train_size = int(raw_data_length * (1 - self._val_split))
        val_size = raw_data_length - train_size

        self.dataset_train, self.dataset_val = random_split(
            raw_data,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(self._seed),
        )

    def prepare_data(self):

        path_train = os.path.join(self._P.TRAINING_PATH, "train.csv")
        path_val = os.path.join(self._P.TRAINING_PATH, "val.csv")

        if os.path.exists(path_train) and os.path.exists(path_val):
            self._prepare_splitted_data(path_train, path_val)

        else:
            self._prepare_no_splitted_data()

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
