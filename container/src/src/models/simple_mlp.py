import os
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from argparse import Namespace


class Model(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._data_module = kwargs.get("data_module", None)
        if self._data_module is not None:
            num_features = self._data_module.num_features
            num_classes = self._data_module.num_classes
            self.save_hyperparameters(
                Namespace(num_features=num_features, num_classes=num_classes)
            )
        else:
            num_features, num_classes = args[0]["num_features"], args[0]["num_classes"]

        self.l1 = torch.nn.Linear(num_features, num_classes)

    def forward(self, x):
        return F.log_softmax(self.l1(x.view(x.size(0), -1)), -1)

    def training_step(self, batch, batch_nb):
        if isinstance(batch, (list, tuple)):
            x, y = batch
        else:
            x = batch[:, 1:]
            y = batch[:, 0].long()
        loss = F.nll_loss(self(x), y)
        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
