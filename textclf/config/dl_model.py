from .base import ConfigBase
from .embedding_layer import EmbeddingLayerConfig, StaticEmbeddingLayerConfig
from .classifier import ClassifierConfig, CNNClassifierConfig


class DLModelConfig(ConfigBase):
    """深度学习模型可以看成两部分:
        1.embedding层:比如word2vec, glove, bert, 或者随机初始化的向量
        2.classifier层：比如TextCNN/全连接层/TextRNN等等
    """
    embedding_layer: EmbeddingLayerConfig = StaticEmbeddingLayerConfig()
    # default model is textcnn
    classifier: ClassifierConfig = CNNClassifierConfig()
