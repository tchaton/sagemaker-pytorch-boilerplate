import os
import os.path as osp
import random
import torch
import hydra
from pytorch_lightning import Trainer
from src.models.mlp import Model
from src.datasets.mnist import MNISTDataset


# These are the paths to where SageMaker mounts interesting things in your container.
PREFIX = '/opt/ml/'
INPUT_PATH = osp.join(PREFIX, 'input/data')
OUTPUT_PATH = osp.join(PREFIX, 'output')
MODEL_PATH = osp.join(PREFIX, 'model')
PARAM_PATH = osp.join(PREFIX, 'input/config/hyperparameters.json')
CHANNEL_NAME = 'training'
TRAINING_PATH = os.path.join(INPUT_PATH, CHANNEL_NAME)

def train(cfg):

    data_module = MNISTDataset(  
            data_dir = '.',
            val_split = 5000,
            num_workers = 16,
            normalize = False,
            seed = 42)
    model = Model()
    trainer = Trainer()
    trainer.fit(model, data_module)
