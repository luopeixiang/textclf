import os

from tabulate import tabulate

from textclf.data.dictionary import Dictionary, LabelDictionary
from textclf.config import PreprocessConfig
from textclf.utils.raw_data import (
    tokenize_file,
    create_tokenizer,
    get_label_prob,
    build_label2id
)


class TextClfRawData(object):
    """对数据进行预处理。分词、构建词典、保存成二进制形式方便读入"""

    def __init__(self, config: PreprocessConfig):
        """
        :param config:预处理的设置
        :type config: PreprocessConfig
        """
        self.config = config
        self.tokenizer = create_tokenizer(config.tokenizer)
        self.train_pairs = tokenize_file(
            os.path.join(config.datadir, config.train_file),
            self.tokenizer
        )
        self.valid_pairs = tokenize_file(
            os.path.join(config.datadir, config.valid_file),
            self.tokenizer
        )
        self.test_pairs = tokenize_file(
            os.path.join(config.datadir, config.test_file),
            self.tokenizer
        )
        self.dictionary = self._build_dictionary()
        self.label2id = build_label2id([label for _, label in self.train_pairs])

    def _build_dictionary(self):
        dictionary = Dictionary()
        for text, _ in self.train_pairs:
            dictionary.add_sentence(text)  # build dict
        dictionary.finalize(
            nwords=self.config.nwords,
            threshold=self.config.min_word_count
        )
        return dictionary

    def describe(self):
        """输出数据的信息：类别分布、字典大小
        """
        headers = [
            "",
            self.config.train_file,
            self.config.valid_file,
            self.config.test_file
        ]
        train_label_prob = get_label_prob([label for _, label in self.train_pairs])
        valid_label_prob = get_label_prob([label for _, label in self.valid_pairs])
        test_label_prob = get_label_prob([label for _, label in self.test_pairs])
        label_table = []
        for label in train_label_prob:
            label_table.append([
                label,
                train_label_prob[label],
                valid_label_prob[label],
                test_label_prob[label]
            ])
        label_table.append([
            "Sum",
            len(self.train_pairs),
            len(self.valid_pairs),
            len(self.test_pairs)
        ])
        print("Label Prob:")
        print(tabulate(label_table, headers, tablefmt="grid", floatfmt=".4f"))

        print(f"Dictionary Size: {len(self.dictionary)}")
