from dataclasses import dataclass, field
from typing import *

from hydra.core.config_store import ConfigStore
from hydra.types import *

defaults = [
    {"model_type": MISSING},
    {"model_name": MISSING},
    {"dataset": MISSING},
    {"trainer": MISSING},
]

@dataclass
class ObjectConf(Dict[str, Any]):
    # class, class method or function name
    target: str = MISSING
    # parameters to pass to target when calling it
    params: Any = field(default_factory=dict)

class DatasetConf(ObjectConf):
    pass

class ModelConf(ObjectConf):
    pass

@dataclass
class Config:
    defaults: List[Any] = field(default_factory=lambda: defaults)

cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(name="model", node=ModelConf)
cs.store(name="data", node=DatasetConf)
