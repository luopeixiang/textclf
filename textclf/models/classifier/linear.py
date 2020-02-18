import torch
import torch.nn as nn

from .base import Classifier
from textclf.config import LinearClassifierConfig


class LinearClassifier(Classifier):
    def __init__(self, config: LinearClassifierConfig):
        super(LinearClassifier, self).__init__(config)

        self.has_hidden_layers = config.hidden_units is not None
        if self.has_hidden_layers:
            hidden_layers = []
            prev_hidden_unit = config.input_size
            for hidden_unit, act, hidden_dp in zip(
                config.hidden_units,
                config.activations,
                config.hidden_dropouts
            ):
                hidden_layers.append(nn.Linear(prev_hidden_unit, hidden_unit))
                hidden_layers.append(getattr(nn, act)())
                hidden_layers.append(nn.Dropout(hidden_dp))
                prev_hidden_unit = hidden_unit
            self.hidden_layers = nn.Sequential(*hidden_layers)

        assert config.pool_method in ["first", "mean"]
        self.pool_method = config.pool_method
        self.emb_dropout = nn.Dropout(config.embedding_dropout)
        last_input_size = hidden_unit if self.has_hidden_layers else config.input_size
        self.output_layer = nn.Linear(last_input_size, config.output_size)

    def forward(self, embedding, seq_lens):

        if self.pool_method == "first":
            hidden = embedding[:, 0, :]  # [batch_size, hidden_size]
        elif self.pool_method == "mean":
            hidden = torch.stack([
                emb[:seq_len].mean(dim=0) for emb, seq_len in zip(embedding, seq_lens)
            ], dim=0)

        if self.has_hidden_layers:
            hidden = self.hidden_layers(hidden)
        return self.output_layer(hidden)
