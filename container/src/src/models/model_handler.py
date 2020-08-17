import os
import os.path as osp
import random
import torch
import hydra
import pytorch_lightning as pl
from src.paths import Paths


class ModelHandler(object):

    model = None  # Where we keep the model when it's loaded

    def __init__(self, opt, P):
        self._initialized = False
        try:
            self._opt = opt
            self._P = P
            self.model = pl.LightningModule.load_from_checkpoint(P.MODEL_PATH)
            self.model.eval()
            self.model.freeze()
            self._initialized = True
        except Exception as e:
            print(e)
            exit()

    @property
    def initialized(self):
        return self._initialized

    def predict(self, input):
        return self.model(input)
