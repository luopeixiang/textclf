from typing import Optional

from .base import ConfigBase
from textclf.data.dictionary import Dictionary


class EmbeddingLayerConfig(ConfigBase):
    pass


class StaticEmbeddingLayerConfig(EmbeddingLayerConfig):
    """静态词向量：word2vec, glove 或者随机初始化的向量"""
    dim: int = 300
    method: str = "random"  # choice: random/pretrained

    # Specify pretrained model path, valid when methed is pretrained
    # The pre-trained vector files are in text format.
    # Each line contains a word and its vector. Each value is separated by space.
    pretrained_path: Optional[str] = None

    # 字典，无效指定，运行时确定
    dictionary: Optional[Dictionary] = None


class BertEmbeddingLayerConfig(EmbeddingLayerConfig):
    # the hidden size of hidden 如果是base则为768,如果是large，则为1024
    dim: int = 768
    model_dir: str = "bert_pretrain"
