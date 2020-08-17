import os
import torch
from torch.nn import functional as F
import pytorch_lightning as pl
from argparse import Namespace


class Model(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        self.l1 = torch.nn.Linear(kwargs["num_features"], kwargs["num_classes"])

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
