from typing import List
import sys
import json

import textclf.config as config
from textclf.config import (
    PreprocessConfig,
    DLTrainerConfig,
    DLTesterConfig,
    MLTrainerConfig,
    MLTesterConfig
)
from textclf.config.base import ConfigBase


def json_to_configdict(json_data):
    res = {}
    for k in json_data:  # 如果含有class，需要将其转化为类
        obj = json_data[k]
        if type(obj) == dict and "__class__" in obj:
            class_name = obj["__class__"]
            res[k] = getattr(config, class_name)(json_to_configdict(obj["params"]))
        else:
            res[k] = obj
    return res


def config_from_json(json_data):
    class_name = json_data['__class__']
    config_dict = json_to_configdict(json_data["params"])
    return getattr(config, class_name)(config_dict)


def get_instance_name(config, drop_suffix=True):
    name = config.__class__.__name__
    return name[:-6] if drop_suffix else name


def stringfy(config_class, drop_suffix=True):
    name = str(config_class).split('.')[-1]
    return name[:-8] if drop_suffix else name[:-2]


def help_config_dfs(config, options):
    """deep first search for config helping"""
    config_keys, common_keys = split_keys(config)
    for key in common_keys:
        options[key] = getattr(config, key)
    if not config_keys:
        return
    for key in config_keys:
        print(f"正在设置{key}")
        value = getattr(config, key)
        base = value.__class__.__base__
        try:
            from textclf.utils.common import CONFIG_CHOICES
            choices = CONFIG_CHOICES[base]
            choice_id = query_choices_to_user(
                [stringfy(c) for c in choices],
                get_instance_name(value),
                key
            )
            if choice_id == -1:
                choice = value
            else:
                choice = choices[choice_id]()
        except KeyError:
            choice = value

        options[key] = {}
        options[key]["__class__"] = choice.__class__.__name__
        options[key]["params"] = {}
        help_config_dfs(choice, options[key]["params"])


def query_choices_to_user(choices: List[str], default: str, base: str):
    """让用户在choices之中做选择"""
    print(f"{base} 有以下选择(Default: {default}): ")
    for i, choice in enumerate(choices):
        print(f"{i}. {choice}")

    while True:
        choice_id = input("输入您选择的ID (q to quit, enter for default):")
        if choice_id == "q":
            print("Goodbye!")
            sys.exit()
        elif choice_id == "":
            print(f"Chooce default value: {default}")
            return -1

        try:
            choice_id = int(choice_id)
            if choice_id not in range(len(choices)):
                print(f"{choice_id} 不在可选范围内！")
            else:
                print(f"Chooce value {choices[choice_id]}")
                return choice_id
        except ValueError:
            print("请输入整数ID！")


def split_keys(config):
    common_keys = []
    config_keys = []
    for k, v in config.items():
        if isinstance(v, ConfigBase):
            config_keys.append(k)
        else:
            common_keys.append(k)
    return config_keys, common_keys


def help_config_main():
    options = {}
    init_choices = [
        (PreprocessConfig, "预处理的设置"),
        (DLTrainerConfig, "训练深度学习模型的设置"),
        (DLTesterConfig, "测试深度学习模型的设置"),
        (MLTrainerConfig, "训练机器学习模型的设置"),
        (MLTesterConfig, "测试机器学习模型的设置")
    ]
    default = DLTrainerConfig()
    choice_id = query_choices_to_user(
        [(stringfy(c, drop_suffix=False)+'\t'+desc) for c, desc in init_choices],
        get_instance_name(default, drop_suffix=False),
        "Config "
    )
    if choice_id == -1:
        config = default
    else:
        config = init_choices[choice_id][0]()

    help_config_dfs(config, options)
    config_dict = {"__class__": config.__class__.__name__}
    config_dict["params"] = options

    # for debug
    # new_config = config_from_json(config_dict)
    # print(f"您的配置如下所示：\n {new_config}")

    # 是否存入文件
    path = input("输入保存的文件名(Default: config.json): ")
    if not path:
        path = "config.json"

    with open(path, "w") as fd:
        json.dump(config_dict, fd, indent=4)
        print(f"已经将您的配置写入到 {path},"
              "你可以在该文件中查看、修改参数以便后续使用")
    print("Bye!")
