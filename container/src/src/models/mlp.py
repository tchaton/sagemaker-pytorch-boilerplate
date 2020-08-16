import os
import torch
from torch.nn import functional as F
import pytorch_lightning as pl

class Model(pl.LightningModule):

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return F.log_softmax(self.l1(x.view(x.size(0), -1)), -1)

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.nll_loss(self(x), y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)