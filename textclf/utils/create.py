import torch

from textclf.config.scheduler import NoneSchedulerConfig
from .config import get_instance_name


def create_instance(config, asdict=False):
    """create machine learing model by config"""
    from .common import CONFIG_TO_CLASS
    class_name = CONFIG_TO_CLASS[config.__class__.__name__]
    if asdict:
        return class_name(**config.asdict())
    else:
        return class_name(config)


def create_dl_model(embedding_config, dictionary, model_config):
    from .common import CONFIG_TO_CLASS
    class_name = CONFIG_TO_CLASS[model_config.__class__.__name__]
    return class_name(embedding_config, dictionary, model_config)


def create_optimizer(optimizer_config, model, epoch=1):
    """构建优化器"""
    options = optimizer_config.asdict()
    static_epoch = options.pop('static_epoch')
    elr = options.pop('embedding_lr')
    if epoch <= static_epoch:  # 在static_epoch之后，embedding层才开始学习
        elr = 0.
    lr = options.pop('lr')

    params_list = []
    for name, param in model.named_parameters():
        if name.endswith("embedding.weight"):
            params_list.append(dict(params=param, lr=elr))
        else:
            params_list.append(dict(params=param, lr=lr))

    # 对于embedding层以及其他层 采取不一样的学习速度
    optimizer_name = get_instance_name(optimizer_config)
    optimizer = getattr(torch.optim, optimizer_name)(params_list, **options)
    return optimizer


def create_lr_scheduler(scheduler_config, optimizer):
    if isinstance(scheduler_config, NoneSchedulerConfig):
        return None
    options = scheduler_config.asdict()
    scheduler_name = get_instance_name(scheduler_config)
    scheduler = getattr(torch.optim.lr_scheduler, scheduler_name)(optimizer, **options)
    return scheduler
