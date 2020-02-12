import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel

from textclf.config.embedding_layer import (
    EmbeddingLayerConfig,
    StaticEmbeddingLayerConfig,
    BertEmbeddingLayerConfig
)


class EmbeddingLayer(nn.Module):
    def __init__(self, config: EmbeddingLayerConfig):
        """init by specified config"""
        super(EmbeddingLayer, self).__init__()
        self.config = config
        self.dim = config.dim

    def forward(self, tokens):
        raise NotImplementedError


class StaticEmbeddingLayer(EmbeddingLayer):
    def __init__(self, config: StaticEmbeddingLayerConfig):
        """init by specified config"""
        super(StaticEmbeddingLayer, self).__init__(config)
        self.embedding = nn.Embedding(len(self.config.dictionary), self.dim)
        if config.method == "pretrained":
            # load embedding from pretrained
            self._init_embedding_from_file(self.config.pretrained_path)

    def forward(self, tokens):
        return self.embedding(tokens)

    def _init_embedding_from_file(self, filename):
        word2vec = {}
        print(f"Loading pretrained word embedding from {filename}...")
        with open(filename) as f:
            for line in f:
                items = line.strip().split()
                word, vec = items[0], items[1:]
                vec = np.array([float(item) for item in vec])
                word2vec[word] = vec

        for idx, word in enumerate(self.config.dictionary.symbols):
            try:
                vec = word2vec[word]
                self.embedding.weight.data[idx] = torch.from_numpy(vec)
            except KeyError:
                continue


class BertEmbeddingLayer(EmbeddingLayer):
    def __init__(self, config: BertEmbeddingLayerConfig):
        super(BertEmbeddingLayer, self).__init__(config)
        self.embedding = BertModel.from_pretrained(self.config.model_dir)

    def forward(self, tokens):
        outputs = self.embedding(tokens)
        last_hidden_states = outputs[0]
        return last_hidden_states
