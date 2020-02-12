
from abc import ABCMeta, abstractmethod


class Trainer(metaclass=ABCMeta):
    """Trainer基类，规定MLTrainer、DLTrainer的接口"""

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def train(self):
        pass
