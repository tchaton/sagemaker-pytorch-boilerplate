import os
import os.path as osp
import random
import torch
import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from src.paths import Paths


class ModelHandler(object):
    model = None  # Where we keep the model when it's loaded
    datamodule = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            P = Paths("aws")
            print(P)
            trainer = pl.Trainer(resume_from_checkpoint=P.MODEL_CHECKPOINT_PATH)
            cls.model = trainer.model
            cls.model.freeze()
            print(cls.model)
            cls.datamodule = trainer.datamodule
            print(cls.datamodule)
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        clf = cls.get_model()
        return clf(input)
