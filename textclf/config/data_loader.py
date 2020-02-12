
from .base import ConfigBase


class DataLoaderConfig(ConfigBase):
    raw_data_path: str = "textclf.joblib"
    # how many samples per batch to load (default: 64).
    batch_size: int = 32
    max_len: int = 10

    # set to True to have the data reshuffled at every epoch (default: False).
    shuffle: bool = False

    # how many subprocesses to use for data loading.
    # 0 means that the data will be loaded in the main process. (default: 0)
    num_workers: int = 0

    # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
    pin_memory: bool = True

    # set to True to drop the last incomplete batch
    drop_last: bool = True
