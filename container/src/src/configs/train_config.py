from dataclasses import dataclass, field
from typing import Any, List

from hydra.core.config_store import ConfigStore

defaults = [
    {"model_type": "NULL"},
    {"model_name": "NULL"},
    {"dataset": "NULL"},
]

@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
