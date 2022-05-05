from typing import List

import torch

from textclf.data.dictionary import Dictionary


def texts_to_tensor(texts: List[List[str]], dictionary: Dictionary):
    max_len = max(len(t) for t in texts)
    tensor = torch.ones(len(texts), max_len).long() * dictionary.pad()
    for i, text in enumerate(texts):
        tensor[i][:len(text)] = dictionary.tokens_to_tensor(text)
    return tensor


def seqLens_to_mask(seq_lens):
    max_len = seq_lens.max().item()
    arange = torch.arange(max_len).expand(len(seq_lens), max_len).to(seq_lens.device)
    mask = (arange >= seq_lens.unsqueeze(1))
    mask = mask.to(seq_lens.device)
    return mask
