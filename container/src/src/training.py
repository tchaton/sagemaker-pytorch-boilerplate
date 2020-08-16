import os
import os.path as osp
import random
import torch
import hydra
from pytorch_lightning import Trainer
from src.models.mlp import Model
from src.datasets.mnist import MNISTDataset


# These are the paths to where SageMaker mounts interesting things in your container.
PREFIX = "/opt/ml/"
INPUT_PATH = osp.join(PREFIX, "input/data")
OUTPUT_PATH = osp.join(PREFIX, "output")
MODEL_PATH = osp.join(PREFIX, "model")
PARAM_PATH = osp.join(PREFIX, "input/config/hyperparameters.json")
CHANNEL_NAME = "training"
TRAINING_PATH = os.path.join(INPUT_PATH, CHANNEL_NAME)


def train(cfg):

    data_module = hydra.utils.instantiate(cfg.dataset)
    model = hydra.utils.instantiate(cfg.model)
    trainer = Trainer()
    trainer.fit(model, data_module)
