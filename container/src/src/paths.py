import os
import os.path as osp

# These are the paths to where SageMaker mounts interesting things in your container.
LOCAL_PREFIX = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "local_test", "test_dir"
)


class Paths:

    AWS_PREFIX = "/opt/ml/"

    def __init__(self, cfg):
        if cfg.mode == "local":
            self._build(LOCAL_PREFIX)
        else:
            self._build(self.AWS_PREFIX)

    def _build(self, prefix):
        self.INPUT_PATH = osp.join(prefix, "input/data")
        self.OUTPUT_PATH = osp.join(prefix, "output")
        self.MODEL_PATH = osp.join(prefix, "model")
        self.PARAM_PATH = osp.join(prefix, "input/config/hyperparameters.json")
        self.CHANNEL_NAME = "training"
        self.TRAINING_PATH = os.path.join(self.INPUT_PATH, self.CHANNEL_NAME)

