本文件根据[../textclf/config/trainer.py](../textclf/config/trainer.py)自动生成

### MLTrainerConfig



MLTrainerConfig有以下属性：

 | Attribute name   | Type             | Default                    | Description                          |
|------------------|------------------|----------------------------|--------------------------------------|
| vectorizer       | VectorizerConfig | CountVectorizerConfig()    |                                      |
| model            | MLModelConfig    | LogisticRegressionConfig() |                                      |
| raw_data_path    | str              | "textclf.joblib"           |                                      |
| save_dir         | str              | "ckpts/"                   | the dir to save model and vectorizer |



### DLTrainerConfig

Traning config for deep learning model

DLTrainerConfig有以下属性：

 | Attribute name              | Type                      | Default                  | Description                                                                                                                                   |
|-----------------------------|---------------------------|--------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| use_cuda                    | bool                      | True                     | 是否使用GPU                                                                                                                                   |
| epochs                      | int                       | 10                       | : Training epochs                                                                                                                             |
| score_method                | str                       | "accuracy"               | score method 指定保存最优模型的方式如果score_method为accuracy，那么保存验证集上准确率最高的模型如果score_method为loss，那么保存损失最小的模型 |
| ckpts_dir                   | str                       | "ckpts"                  | 指定checkpoints保存的目录                                                                                                                     |
| save_ckpt_every_epoch       | bool                      | True                     | 是否每个epoch都保存ckpt                                                                                                                       |
| random_state                | Optional[int]             | 2020                     | 随机数种子，保证每次结果相同                                                                                                                  |
| state_dict_file             | Optional[str]             | None                     | random_state: Optional[int] = None从state_dict_file指定的断点开始训练state_dict_file: Optional[str] = "./ckpts/1.pt"                          |
| early_stop_after            | Optional[int]             | None                     | : Stop after how many epochs when the eval metric is not improving                                                                            |
| max_clip_norm               | Optional[float]           | None                     | : Clip gradient norm if set                                                                                                                   |
| do_eval                     | bool                      | True                     | : Whether to do evaluation and model selection based on it.                                                                                   |
| load_best_model_after_train | bool                      | True                     | : if do_eval, do we load the best model state dict after training or justuse the latest model state                                           |
| num_batch_to_print          | int                       | 10                       | : Number of samples for print training info.                                                                                                  |
| optimizer                   | OptimizerConfig           | AdamConfig()             | : config for optimizer, used in parameter update                                                                                              |
| scheduler                   | Optional[SchedulerConfig] | NoneSchedulerConfig()    |                                                                                                                                               |
| model                       | DLModelConfig             | DLModelConfig()          | config Classifer                                                                                                                              |
| data_loader                 | DataLoaderConfig          | DataLoaderConfig()       | config data loader                                                                                                                            |
| criterion                   | CriterionConfig           | CrossEntropyLossConfig() | config criterion                                                                                                                              |

