本文件根据[../textclf/config/scheduler.py](../textclf/config/scheduler.py)自动生成

### SchedulerConfig

 无可设置的属性



### NoneSchedulerConfig

Do nothing 无可设置的属性



### ReduceLROnPlateauConfig



ReduceLROnPlateauConfig继承SchedulerConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type   | Default   | Description                                                                                                                                                                                                                                                                    |
|------------------|--------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| mode             | str    | 'min'     | One of min, max.In min mode, lr will be reduced when val loss has stopped decreasing;in max mode it will be reduced when val accuracy has stopped increasing. Default: ‘min’.                                                                                                  |
| factor           | float  | 0.1       | Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.                                                                                                                                                                                         |
| patience         | int    | 10        | Number of epochs with no improvement after which learning rate will be reduced.For example, if patience = 2, then we will ignore the first 2 epochs with no improvement,and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10. |
| verbose          | bool   | True      | If True, prints a message to stdout for each update. Default: True.                                                                                                                                                                                                            |
| threshold        | float  | 1e-4      | Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.                                                                                                                                                                                  |
| threshold_mode   | str    | "rel"     | One of rel, abs. In rel mode, dynamic_threshold = best * (1 + threshold) in ‘max’ mode or best * (1 - threshold)in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best-threshold in min mode.                                                      |
| cooldown         | int    | 0         | Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.                                                                                                                                                                               |
| min_lr           | float  | 0.        | A lower bound on the learning rate of all param groups or each group respectively. Default: 0.                                                                                                                                                                                 |
| eps              | float  | 1e-8      | Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored.                                                                                                                                                              |



### StepLRConfig



StepLRConfig继承SchedulerConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type   | Default   | Description                                                 |
|------------------|--------|-----------|-------------------------------------------------------------|
| step_size        | int    | 10        | Period of learning rate decay.                              |
| gamma            | float  | 0.5       | Multiplicative factor of learning rate decay. Default: 0.1. |



### MultiStepLRConfig



MultiStepLRConfig继承SchedulerConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type      | Default     | Description                                                 |
|------------------|-----------|-------------|-------------------------------------------------------------|
| milestones       | List[int] | [5, 10, 15] | List of epoch indices. Must be increasing.                  |
| gamma            | float     | 0.5         | Multiplicative factor of learning rate decay. Default: 0.1. |

