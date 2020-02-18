from typing import Tuple

from .base import ConfigBase

"""参考https://pytorch.org/docs/stable/optim.html"""


class OptimizerConfig(ConfigBase):
    # learning rate for embedding layer
    embedding_lr: float = 1e-3
    # learning rate for other layer
    lr: float = 1e-3

    # embedding层在前static_epoch将不会进行训练
    # 在static_epoch之后，embedding将会以embedding_lr作为学习率进行训练
    static_epoch: int = 0


class AdamConfig(OptimizerConfig):
    """
    Adam algorithm.
    It has been proposed in Adam: A Method for Stochastic Optimization.
    """

    # coefficients used for computing running averages of gradient
    # and its square(default: (0.9, 0.999))
    betas: Tuple[float, float] = (0.9, 0.999)

    # term added to the denominator to improve numerical stability(default: 1e-8)
    eps: float = 1e-8

    # weight decay(L2 penalty)(default: 0)
    weight_decay: float = 0.

    # whether to use the AMSGrad variant of this algorithm from the paper
    # On the Convergence of Adam and Beyond
    amsgrad: bool = False


class AdadeltaConfig(OptimizerConfig):
    """
    Adadelta algorithm.
    It has been proposed in ADADELTA: An Adaptive Learning Rate Method.
    """

    # coefficient used for computing a running average of squared gradients(default: 0.9)
    rho: float = 0.9

    # term added to the denominator to improve numerical stability(default: 1e-6)
    eps: float = 1e-6

    # weight decay(L2 penalty)(default: 0)
    weight_decay: float = 0.


class AdagradConfig(OptimizerConfig):
    """
    Adagrad algorithm.
    It has been proposed in Adaptive Subgradient Methods for
        Online Learning and Stochastic Optimization.
    """

    # learning rate decay (default: 0)
    lr_decay: float = 0.0

    # weight decay (L2 penalty) (default: 0)
    weight_decay: float = 0.0

    # term added to the denominator to improve numerical stability (default: 1e-10)
    eps: float = 1e-10


class AdamWConfig(OptimizerConfig):
    """
    AdamW algorithm.
    The original Adam algorithm was proposed in Adam:
        A Method for Stochastic Optimization.
    The AdamW variant was proposed in Decoupled Weight Decay Regularization.
    """

    # coefficients used for computing running averages of gradient and its square(default: (0.9, 0.999))
    betas: Tuple[float, float] = (0.9, 0.999)

    # term added to the denominator to improve numerical stability(default: 1e-8)
    eps: float = 1e-8

    # weight decay coefficient(default: 1e-2)
    weight_decay: float = 1e-2

    # whether to use the AMSGrad variant of this algorithm from the paper
    # On the  Convergence of Adam and Beyond(default: False)
    amsgrad: bool = False


class AdamaxConfig(OptimizerConfig):

    """
    Adamax algorithm (a variant of Adam based on infinity norm).
    It has been proposed in Adam: A Method for Stochastic Optimization.
    """

    # coefficients used for computing running averages of gradient and its square(default: (0.9, 0.999))
    betas: Tuple[float, float] = (0.9, 0.999)

    # term added to the denominator to improve numerical stability(default: 1e-8)
    eps: float = 1e-8

    # weight decay coefficient(default: 1e-2)
    weight_decay: float = 1e-2


class ASGDConfig(OptimizerConfig):
    """
    Averaged Stochastic Gradient Descent.
    It has been proposed in Acceleration of stochastic approximation by averaging.
    """

    # decay term (default: 1e-4)
    lambd: float = 1e-4

    # power for eta update (default: 0.75)
    alpha: float = 0.75

    # point at which to start averaging (default: 1e6)
    t0: float = 1e6

    # weight decay (L2 penalty) (default: 0)
    weight_decay: float = 0


class RMSpropConfig(OptimizerConfig):
    """
    RMSprop algorithm.
    Proposed by G. Hinton in his course.
    """

    # momentum factor(default: 0)
    momentum: float = 0.0

    # smoothing constant(default: 0.99)
    alpha: float = 0.99

    # term added to the denominator to improve numerical stability(default: 1e-8)
    eps: float = 1e-8

    # if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
    centered: bool = False
    # weight decay(L2 penalty)(default: 0)
    weight_decay: float = 0.


class RpropConfig(OptimizerConfig):
    """
    The resilient backpropagation algorithm.
    """

    # pair of (etaminus, etaplis), that are multiplicative increase and decrease factors (default: (0.5, 1.2))
    etas: Tuple[float, float] = (0.5, 1.2)

    # a pair of minimal and maximal allowed step sizes (default: (1e-6, 50))
    step_sizes: Tuple[float, float] = (1e-6, 50)


class SGDConfig(OptimizerConfig):
    """
    stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from
    On the importance of initialization and momentum in deep learning.
    """

    # momentum factor (default: 0)
    momentum: float = 0.

    # weight decay (L2 penalty) (default: 0)
    weight_decay: float = 0.

    # dampening for momentum (default: 0)
    dampening: float = 0.
    # enables Nesterov momentum (default: False)
    nesterov: bool = False
