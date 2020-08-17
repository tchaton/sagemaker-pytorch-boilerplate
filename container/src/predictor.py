#!/usr/bin/env python

# A sample training component that trains a simple scikit-learn decision tree model.
# This implementation works in File mode and makes no assumptions about the input file names.
# Input is specified as CSV with a data point in each row and the labels in the first column.

from __future__ import print_function

import os
import hydra

os.environ["HYDRA_FULL_ERROR"] = "1"
import os.path as osp
import json
import pickle
import sys
import traceback
from omegaconf import DictConfig
import random
import torch
from src.datasets import *
from src.paths import Paths
from src.models.model_handler import ModelHandler
from src.configs.train_config import *
from src.app import app


# https://hydra.cc/docs/next/tutorials/structured_config/hierarchical_static_config
@hydra.main(config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    P = Paths(cfg)
    model_handler = ModelHandler.load_model(osp.join(P.MODEL_PATH, "model.ckpt"))


if __name__ == "__main__":
    my_app()
