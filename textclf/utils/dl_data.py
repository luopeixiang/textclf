from typing import List

import torch

from textclf.data.dictionary import Dictionary


def texts_to_tensor(texts: List[List[str]], dictionary: Dictionary):
    max_len = max(len(t) for t in texts)
    tensor = torch.ones(len(texts), max_len).long() * dictionary.pad()
    for i, text in enumerate(texts):
        tensor[i][:len(text)] = dictionary.tokens_to_tensor(text)
    return tensor
