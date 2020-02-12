from typing import Optional, List

from .base import ConfigBase


class CriterionConfig(ConfigBase):
    # 是否使用GPU，运行时确定
    use_cuda: Optional[bool] = None


class CrossEntropyLossConfig(CriterionConfig):

    # a manual rescaling weight given to each class. If given, has to be a Tensor of size C
    weight: Optional[List[float]] = None

    # Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
    # 'none': no reduction will be applied,
    # 'mean': the sum of the output will be divided by the number of elements in the output,
    # 'sum': the output will be summed.
    # Default: 'mean'
    reduction: str = 'mean'

    # 是否应用label smooth
    label_smooth_eps: Optional[float] = None


class FocalLossConfig(CriterionConfig):
    """Focal Loss for Dense Object Detection, https://arxiv.org/abs/1708.02002"""
    gamma = 2.0
    alpha = 0.25
    reduction: str = "mean"
