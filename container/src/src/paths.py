import os
import os.path as osp

# These are the paths to where SageMaker mounts interesting things in your container.
LOCAL_PREFIX = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "..", "local_test", "test_dir"
)

CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "..", "conf", "config.yaml"
)


class Paths:

    AWS_PREFIX = "/opt/ml/"
    CONFIG_PATH = "/opt/program/conf/config.yaml"
    MODEL_NAME = "model.ckpt"
    TRAINER_NAME = "trainer.ckpt"

    def __init__(self, mode):
        self.MODE = mode
        if mode == "local":
            self._build(LOCAL_PREFIX, CONFIG_PATH)
        else:
            self._build(self.AWS_PREFIX, self.CONFIG_PATH)

    def _build(self, prefix, config_path):
        self.INPUT_PATH = osp.join(prefix, "input/data")
        self.OUTPUT_PATH = osp.join(prefix, "output")
        self.MODEL_PATH = osp.join(prefix, "model")
        self.PARAM_PATH = osp.join(prefix, "input/config/hyperparameters.json")
        self.CHANNEL_NAME = "training"
        self.TRAINING_PATH = os.path.join(self.INPUT_PATH, self.CHANNEL_NAME)
        self.MODEL_CHECKPOINT_PATH = os.path.join(self.MODEL_PATH, self.MODEL_NAME)
        self.TRAINER_CHECKPOINT_PATH = os.path.join(self.MODEL_PATH, self.TRAINER_NAME)
        self.CONFIG_PATH = config_path

    def __repr__(self):
        msg = ""
        for key, item in self.__dict__.items():
            if "__" not in key:
                msg += "{}: {} \n".format(key, item)
        return msg

