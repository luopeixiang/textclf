from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

from textclf.trainer import MLTrainer, DLTrainer  # 循环import
from textclf.models.classifier import (
    CNNClassifier,
    LinearClassifier,
    RNNClassifier,
    RCNNClassifier,
    DRNNClassifier,
    DPCNNClassifier
)
from textclf.models.embedding_layer import (
    EmbeddingLayerConfig,
    StaticEmbeddingLayer,
    BertEmbeddingLayer
)
from textclf.models.dl_model import DLModel
from textclf.tester import MLTester, DLTester
from textclf.trainer.criterion import CrossEntropyLoss, FocalLoss
from textclf.config.optimizer import OptimizerConfig
from textclf.config.classifier import ClassifierConfig
from textclf.config.criterion import CriterionConfig
from textclf.config.scheduler import SchedulerConfig
from textclf.config.vectorizer import VectorizerConfig
from textclf.config.ml_model import MLModelConfig


CONFIG_TO_CLASS = {
    "CountVectorizerConfig": CountVectorizer,
    "TfidfVectorizerConfig": TfidfVectorizer,
    "LogisticRegressionConfig": LogisticRegression,
    "LinearSVMConfig": SGDClassifier,
    "MLTrainerConfig": MLTrainer,
    "DLTrainerConfig": DLTrainer,
    "CNNClassifierConfig": CNNClassifier,
    "RNNClassifierConfig": RNNClassifier,
    "RCNNClassifierConfig": RCNNClassifier,
    "DRNNClassifierConfig": DRNNClassifier,
    "DPCNNClassifierConfig": DPCNNClassifier,
    "LinearClassifierConfig": LinearClassifier,
    "StaticEmbeddingLayerConfig": StaticEmbeddingLayer,
    "BertEmbeddingLayerConfig": BertEmbeddingLayer,
    "DLModelConfig": DLModel,
    "MLTesterConfig": MLTester,
    "DLTesterConfig": DLTester,
    "CrossEntropyLossConfig": CrossEntropyLoss,
    "FocalLossConfig": FocalLoss,
}

CONFIG_CHOICES = {
    VectorizerConfig: VectorizerConfig.__subclasses__(),
    OptimizerConfig: OptimizerConfig.__subclasses__(),
    ClassifierConfig: ClassifierConfig.__subclasses__(),
    CriterionConfig: CriterionConfig.__subclasses__(),
    SchedulerConfig: SchedulerConfig.__subclasses__(),
    EmbeddingLayerConfig: EmbeddingLayerConfig.__subclasses__(),
    MLModelConfig: MLModelConfig.__subclasses__()
}
