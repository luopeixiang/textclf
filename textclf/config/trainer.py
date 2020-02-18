from typing import Any, Iterable, List, Optional, Tuple

from .vectorizer import (
    VectorizerConfig,
    CountVectorizerConfig,
    TfidfVectorizerConfig
)
from .ml_model import MLModelConfig, LogisticRegressionConfig
from .dl_model import DLModelConfig
from .optimizer import OptimizerConfig, AdamConfig
from .scheduler import SchedulerConfig
from .data_loader import DataLoaderConfig
from .base import ConfigBase
from .criterion import CriterionConfig, CrossEntropyLossConfig
from .scheduler import NoneSchedulerConfig


class MLTrainerConfig(ConfigBase):
    vectorizer: VectorizerConfig = CountVectorizerConfig()
    model: MLModelConfig = LogisticRegressionConfig()
    raw_data_path: str = "textclf.joblib"

    # the dir to save model and vectorizer
    save_dir: str = "ckpts/"


class DLTrainerConfig(ConfigBase):
    """Traning config for deep learning model"""
    # 是否使用GPU
    use_cuda: bool = True
    #: Training epochs
    epochs: int = 10
    # score method 指定保存最优模型的方式
    # 如果score_method为accuracy，那么保存验证集上准确率最高的模型
    # 如果score_method为loss，那么保存损失最小的模型
    score_method: str = "accuracy"
    # 指定checkpoints保存的目录
    ckpts_dir: str = "ckpts"
    # 是否每个epoch都保存ckpt
    save_ckpt_every_epoch: bool = True
    # 随机数种子，保证每次结果相同
    random_state: Optional[int] = 2020
    # random_state: Optional[int] = None

    # 从state_dict_file指定的断点开始训练
    # state_dict_file: Optional[str] = "./ckpts/1.pt"
    state_dict_file: Optional[str] = None

    #: Stop after how many epochs when the eval metric is not improving
    early_stop_after: Optional[int] = None
    #: Clip gradient norm if set
    max_clip_norm: Optional[float] = None
    #: Whether to do evaluation and model selection based on it.
    do_eval: bool = True
    #: if do_eval, do we load the best model state dict after training or just
    # use the latest model state
    load_best_model_after_train: bool = True

    #: Number of samples for print training info.
    num_batch_to_print: int = 10

    #: config for optimizer, used in parameter update
    optimizer: OptimizerConfig = AdamConfig()
    scheduler: Optional[SchedulerConfig] = NoneSchedulerConfig()

    # config Classifer
    model: DLModelConfig = DLModelConfig()

    # config data loader
    data_loader: DataLoaderConfig = DataLoaderConfig()
    # config criterion
    criterion: CriterionConfig = CrossEntropyLossConfig()
