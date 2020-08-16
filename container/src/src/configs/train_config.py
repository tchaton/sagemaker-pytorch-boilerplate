from dataclasses import dataclass, field
from typing import *

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

defaults = [
    # An error will be raised if the user forgets to specify `db=...`
    {"model": MISSING},
    {"dataset": MISSING},
]


@dataclass
class Trainer:
    max_epochs: int = 100
    gpus: int = 0


@dataclass
class ObjectConf(Dict[str, Any]):
    # class, class method or function name
    target: str = MISSING
    # parameters to pass to target when calling it
    params: Any = field(default_factory=dict)


@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)
    # Hydra will populate this field based on the defaults list
    model: Any = MISSING
    dataset: Any = MISSING
    trainer: Trainer = Trainer()
    mode: str = "local"


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="model", name="mlp_template", node=ObjectConf)
cs.store(group="model", name="new_model", node=ObjectConf)
cs.store(group="dataset", name="mnist_template", node=ObjectConf)
cs.store(group="trainer", name="default", node=Trainer)

