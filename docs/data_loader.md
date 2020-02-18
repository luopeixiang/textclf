本文件根据[../textclf/config/data_loader.py](../textclf/config/data_loader.py)自动生成

### DataLoaderConfig



DataLoaderConfig有以下属性：

 | Attribute name   | Type   | Default          | Description                                                                                                          |
|------------------|--------|------------------|----------------------------------------------------------------------------------------------------------------------|
| raw_data_path    | str    | "textclf.joblib" |                                                                                                                      |
| batch_size       | int    | 32               | how many samples per batch to load (default: 64).                                                                    |
| max_len          | int    | 10               |                                                                                                                      |
| shuffle          | bool   | False            | set to True to have the data reshuffled at every epoch (default: False).                                             |
| num_workers      | int    | 0                | how many subprocesses to use for data loading.0 means that the data will be loaded in the main process. (default: 0) |
| pin_memory       | bool   | True             | If True, the data loader will copy Tensors into CUDA pinned memory before returning them.                            |
| drop_last        | bool   | True             | set to True to drop the last incomplete batch                                                                        |

