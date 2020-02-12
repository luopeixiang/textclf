import os

from .base import ConfigBase


class PreprocessConfig(ConfigBase):
    train_file: str = "train.csv"
    valid_file: str = "valid.csv"
    test_file: str = "test.csv"
    datadir: str = "dataset"
    # Possible choices: space, char, nltk, jiaba
    tokenizer: str = "char"

    # number of words to retain, -1 表示不限制字典的大小
    nwords: int = -1
    # min word count in dict
    min_word_count: int = 1
