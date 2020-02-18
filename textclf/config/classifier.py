from typing import Union, Optional, List

from .base import ConfigBase
from .components import RNNConfig


class ClassifierConfig(ConfigBase):
    # input_size是embedding layer层的维度
    input_size: Optional[int] = None
    # output_size是输出空间的大小,应该与标签数量相对应
    output_size: Optional[int] = None
    # input_size 和 output_size
    # 不需要指定，model在创建的时候会根据标签数量确定


class CNNClassifierConfig(ClassifierConfig):
    kernel_sizes: List[int] = [2, 3, 4]
    num_kernels: int = 256
    top_k_max_pooling: int = 1  # max top-k pooling.
    hidden_layer_dropout: float = 0.5


class LinearClassifierConfig(ClassifierConfig):
    # first or mean
    # if first, use first token's embedidng as Linear layer input
    # if mean, average sequences embedding as linear layer input
    pool_method: str = "first"

    # dropout probability of the embedding layer
    embedding_dropout: float = 0.1

    # 隐藏层设置
    hidden_units: Optional[List[int]] = None
    activations: Optional[List[str]] = None
    hidden_dropouts: Optional[List[float]] = None


class RNNClassifierConfig(ClassifierConfig):

    # config for RNN layer
    rnn_config: RNNConfig = RNNConfig()
    # If True, use attention mechanism to caluate output state
    use_attention: bool = True

    # dropout probability on context
    dropout: float = 0.2


class RCNNClassifierConfig(ClassifierConfig):
    # config for RNN layer
    rnn_config: RNNConfig = RNNConfig()

    # size of  latent semantic vector
    # Refer to Equation 4 of the original paper
    # "Recurrent Convolutional Neural Networks for Text Classification"
    semantic_units: int = 512


class DRNNClassifierConfig(ClassifierConfig):
    # config for RNN layer
    rnn_config: RNNConfig = RNNConfig()

    # The dropout probability  applied  in drnn
    # input and output layers,
    dropout: float = 0.2

    # The window size for rnn
    window_size: int = 10


class DPCNNClassifierConfig(ClassifierConfig):
    # kernel size.
    kernel_size: int = 3
    # stride of pooling.
    pooling_stride: int = 2
    # number of kernels.
    num_kernels: int = 16
    # number of blocks for DPCNN.
    blocks: int = 2

    # dropout probability on convolution features
    dropout: float = 0.2
