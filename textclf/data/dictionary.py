# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# import os
from collections import Counter
from typing import List, Optional

import torch

from textclf.utils.tokenizer import tokenize_line


class Dictionary(object):
    """A mapping from symbols to consecutive integers"""

    def __init__(
        self,
        pad="<pad>",
        unk="<unk>",
        extra_special_symbols=None,
    ):
        self.unk_word, self.pad_word = unk, pad
        self.symbols = []
        self.count = []
        self.indices = {}
        self.pad_index = self.add_symbol(pad)
        self.unk_index = self.add_symbol(unk)
        if extra_special_symbols:
            for s in extra_special_symbols:
                self.add_symbol(s)
        self.nspecial = len(self.symbols)

    def __eq__(self, other):
        return self.indices == other.indices

    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        return self.unk_word

    def __len__(self):
        """Returns the number of symbols in the dictionary"""
        return len(self.symbols)

    def __contains__(self, sym):
        return sym in self.indices

    def index(self, sym):
        """Returns the index of the specified symbol"""
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unk_index

    def string(self, tensor, bpe_symbol=None, escape_unk=False):
        """Helper for converting a tensor of token indices to a string.

        Can optionally remove BPE symbols or escape <unk> words.
        """
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(self.string(t, bpe_symbol, escape_unk) for t in tensor)

        def token_string(i):
            if i == self.unk():
                return self.unk_string(escape_unk)
            else:
                return self[i]

        if hasattr(self, "bos_index"):
            sent = " ".join(
                token_string(i)
                for i in tensor
                if (i != self.eos()) and (i != self.bos())
            )
        else:
            sent = " ".join(token_string(i) for i in tensor if i != self.eos())
        # return data_utils.process_bpe_symbol(sent, bpe_symbol)
        return sent

    def tokens_to_tensor(self, tokens: List[str], max_len: Optional[int]=None):
        """tokens_to_tensor
        :param tokens:
        :type tokens: List[str],需要转化为id 的tokens
        :param max_len: 如果给定，那么当tokens大于max_len时会进行截断，小于则会填充
        :type max_len: Optional
        """
        tensor = torch.ones(len(tokens)).long()
        for i, token in enumerate(tokens):
            tensor[i] = self.index(token)

        if max_len is not None:
            if len(tensor) >= max_len:
                tensor = tensor[:max_len]
            else:
                tensor = torch.cat(
                    [tensor, torch.ones(max_len-len(tensor)).long()*self.pad()])
        return tensor

    def unk_string(self, escape=False):
        """Return unknown string, optionally escaped as: <<unk>>"""
        if escape:
            return "<{}>".format(self.unk_word)
        else:
            return self.unk_word

    def add_symbol(self, word, n=1):
        """Adds a word to the dictionary"""
        if word in self.indices:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx

    def update(self, new_dict):
        """Updates counts from new dictionary."""
        for word in new_dict.symbols:
            idx2 = new_dict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + new_dict.count[idx2]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(new_dict.count[idx2])

    def finalize(self, threshold=-1, nwords=-1):
        """Sort symbols by frequency in descending order, ignoring special ones.

        Args:
            - threshold defines the minimum word count
            - nwords defines the total number of words in the final dictionary,
                including special symbols
        """
        if nwords <= 0:
            nwords = len(self)

        new_indices = dict(zip(self.symbols[: self.nspecial], range(self.nspecial)))
        new_symbols = self.symbols[: self.nspecial]
        new_count = self.count[: self.nspecial]

        c = Counter(
            dict(
                sorted(zip(self.symbols[self.nspecial:], self.count[self.nspecial:]))
            )
        )
        for symbol, count in c.most_common(nwords - self.nspecial):
            if count >= threshold:
                new_indices[symbol] = len(new_symbols)
                new_symbols.append(symbol)
                new_count.append(count)
            else:
                break

        assert len(new_symbols) == len(new_indices)

        self.count = list(new_count)
        self.symbols = list(new_symbols)
        self.indices = new_indices

    def pad(self):
        """Helper to get index of pad symbol"""
        return self.pad_index

    def unk(self):
        """Helper to get index of unk symbol"""
        return self.unk_index

    @classmethod
    def load(cls, f):
        """Loads the dictionary from a text file with the format:

        ```
        <symbol0> <count0>
        <symbol1> <count1>
        ...
        ```
        """
        d = cls()
        d.add_from_file(f)
        return d

    def add_from_file(self, filename: str):
        """
        Loads a pre-existing dictionary from a text file and adds its symbols
        to this instance.
        """
        with open(filename) as f:
            lines = f.readlines()
            indices_start_line = self._load_meta(lines)
            for line in lines[indices_start_line:]:
                idx = line.rfind(" ")
                if idx == -1:
                    raise ValueError(
                        "Incorrect dictionary format, expected '<token> <cnt>'"
                    )
                word = line[:idx]
                count = int(line[idx + 1:])
                self.indices[word] = len(self.symbols)
                self.symbols.append(word)
                self.count.append(count)

    def save(self, filename):
        """Stores dictionary into a text file"""
        c = Counter(
            dict(
                sorted(zip(self.symbols[self.nspecial:], self.count[self.nspecial:]))
            )
        )
        with open(filename, 'w') as f:
            for symbol, count in c.most_common():
                f.write(f"{symbol} {count}\n")
        print(f"save dict to {filename}!")

    def dummy_sentence(self, length):
        t = torch.Tensor(length).uniform_(self.nspecial + 1, len(self)).long()
        t[-1] = self.eos()
        return t

    def encode_line(
        self,
        line,
        line_tokenizer=tokenize_line,
        add_if_not_exist=True,
        consumer=None,
        append_eos=True,
        reverse_order=False,
    ):
        words = line_tokenizer(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = self.add_symbol(word)
            else:
                idx = self.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = self.eos_index
        return ids

    def add_sentence(self, sentence):
        for symbol in sentence.split():
            self.add_symbol(symbol)


class LabelDictionary(object):
    """A mapping from label to consecutive integers"""

    def __init__(self, label_list):
        counter = Counter(label_list)
        self.label2id = {}
        self.id2label = {}
        for label in counter:
            id_ = len(self.label2id)
            self.label2id[label] = id_
            self.id2label[id_] = label
