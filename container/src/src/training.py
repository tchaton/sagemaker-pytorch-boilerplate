import os
import os.path as osp
import random
import torch
import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from src.datasets import *
from src.paths import Paths


def train(cfg):

    P = Paths(cfg.mode)
    print(P)

    data_module = hydra.utils.instantiate(cfg.dataset, P=P)
    model = hydra.utils.instantiate(cfg.model, **data_module.hyper_parameters)

    checkpoint_callback = ModelCheckpoint(filepath=P.MODEL_PATH,)

    trainer = Trainer(checkpoint_callback=checkpoint_callback, max_epochs=2)
    trainer.fit(model, data_module)
    trainer.save_checkpoint(P.TRAINER_CHECKPOINT_PATH)
    import pdb

    pdb.set_trace()
    new_model = model.__class__.load_from_checkpoint(P.TRAINER_CHECKPOINT_PATH)
    print(new_model)
    print("Training complete.")
