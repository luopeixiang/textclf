from typing import List

from .base import ConfigBase


class SchedulerConfig(ConfigBase):
    pass


class NoneSchedulerConfig(SchedulerConfig):
    """Do nothing"""
    pass


class ReduceLROnPlateauConfig(SchedulerConfig):
    # One of min, max.
    # In min mode, lr will be reduced when val loss has stopped decreasing;
    # in max mode it will be reduced when val accuracy has stopped increasing. Default: ‘min’.
    mode: str = 'min'

    # Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
    factor: float = 0.1

    # Number of epochs with no improvement after which learning rate will be reduced.
    # For example, if patience = 2, then we will ignore the first 2 epochs with no improvement,
    # and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
    patience: int = 10
    # If True, prints a message to stdout for each update. Default: True.
    verbose: bool = True

    # Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
    threshold: float = 1e-4

    # One of rel, abs. In rel mode, dynamic_threshold = best * (1 + threshold) in ‘max’ mode or best * (1 - threshold)
    # in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best-threshold in min mode.
    threshold_mode: str = "rel"

    # Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
    cooldown: int = 0

    # A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
    min_lr: float = 0.

    # Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored.
    eps: float = 1e-8


class StepLRConfig(SchedulerConfig):
    # Period of learning rate decay.
    step_size: int = 10
    # Multiplicative factor of learning rate decay. Default: 0.1.
    gamma: float = 0.5


class MultiStepLRConfig(SchedulerConfig):
    # List of epoch indices. Must be increasing.
    milestones: List[int] = [5, 10, 15]
    # Multiplicative factor of learning rate decay. Default: 0.1.
    gamma: float = 0.5
