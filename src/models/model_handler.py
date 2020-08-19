import os
import os.path as osp
import random
import torch
import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from src.paths import Paths
from src.configs.train_config import *
from hydra.experimental import compose, initialize


class ModelHandler(object):
    model = None  # Where we keep the model when it's loaded
    data_module_cls = None

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            initialize(
                config_path="../../conf", strict=True,
            )
            cfg = compose("config.yaml")
            P = Paths("aws")
            model_cls = hydra.utils.get_class(cfg.model.target)
            cls.model = model_cls.load_from_checkpoint(P.TRAINER_CHECKPOINT_PATH)
            cls.model.freeze()
            cls.data_module_cls = hydra.utils.get_class(cfg.dataset.target)
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.
        Args:
            input (a pandas dataframe): The data on which to do the predictions. There will be
                one prediction per row in the dataframe"""
        model = cls.get_model()
        input = cls.data_module_cls.process(input)
        return model.model_fn(input).numpy()
