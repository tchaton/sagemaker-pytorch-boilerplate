import os
import os.path as osp

import hydra
from pytorch_lightning import Trainer


# These are the paths to where SageMaker mounts interesting things in your container.
PREFIX = '/opt/ml/'
INPUT_PATH = osp.join(PREFIX, 'input/data')
OUTPUT_PATH = osp.join(PREFIX, 'output')
MODEL_PATH = osp.join(PREFIX, 'model')
PARAM_PATH = osp.join(PREFIX, 'input/config/hyperparameters.json')
CHANNEL_NAME = 'training'
TRAINING_PATH = os.path.join(INPUT_PATH, CHANNEL_NAME)

def train(cfg):
    
    trainer = Trainer(cfg.trainer)
