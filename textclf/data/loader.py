from typing import Tuple, Iterable
from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

from textclf.utils.dl_data import texts_to_tensor
from .dictionary import Dictionary


class TextClfDataset(Dataset):
    def __init__(self, pairs: Iterable[Tuple[str, int]]):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def build_loader(pairs, dictionary_or_tokenizer, label2id, config):
    pairs = [(text, label2id[label]) for text, label in pairs]
    if isinstance(dictionary_or_tokenizer, Dictionary):
        col_fn = partial(collate_fn, dictionary_or_tokenizer, config.max_len)
    elif isinstance(dictionary_or_tokenizer, BertTokenizer):
        col_fn = partial(bert_collate_fn, dictionary_or_tokenizer, config.max_len)

    loader = DataLoader(
        dataset=TextClfDataset(pairs),
        collate_fn=col_fn,
        batch_size=config.batch_size,
        shuffle=config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=config.drop_last
    )
    return loader


def collate_fn(dictionary: Dictionary, max_len: int, pairs: Iterable[Tuple[str, int]]):
    """collate function for data loader"""
    pairs = [(text.split()[:max_len], label) for text, label in pairs]
    texts, labels = zip(*pairs)
    text_lens = torch.LongTensor([len(text) for text in texts])
    text_tensor = texts_to_tensor(texts, dictionary)
    labels = torch.LongTensor(labels)
    return text_tensor, text_lens, labels


def bert_collate_fn(
    tokenizer: BertTokenizer,
    max_len: int,
    pairs: Iterable[Tuple[str, int]]
):
    pairs = [(text.split()[:max_len], label) for text, label in pairs]
    texts, labels = zip(*pairs)
    labels = torch.LongTensor(labels)
    # +1 for [CLS] token
    text_lens = torch.LongTensor([len(text)+1 for text in texts])
    max_len = text_lens.max().item()
    ids = torch.ones(len(texts), max_len).long() * tokenizer.pad_token_id
    for i, text in enumerate(texts):
        ids[i][:len(text)+1] = torch.LongTensor(
            tokenizer.encode(text, add_special_tokens=True)[:-1])
    return ids, text_lens, labels
