import torch
import torch.nn as nn
import torch.nn.functional as F

from textclf.config import RCNNClassifierConfig
from .base import Classifier
from .components import RNN


class RCNNClassifier(Classifier):
    def __init__(self, config: RCNNClassifierConfig):
        super(RCNNClassifier, self).__init__(config)
        self.semantic_layer = nn.Linear(
            config.rnn_config.hidden_size * 2 + config.input_size,
            config.semantic_units
        )

        rnn_config = config.rnn_config
        rnn_config.input_size = config.input_size
        self.rnn = RNN(rnn_config)
        self.output_layer = nn.Linear(
            config.semantic_units, config.output_size)

    def forward(self, embedding, seq_lens):
        outputs, _ = self.rnn(embedding, seq_lens)
        outputs = torch.cat((embedding, outputs), 2)
        outputs = F.tanh(self.semantic_layer(outputs))
        outputs = outputs.permute(0, 2, 1)
        outputs = F.max_pool1d(outputs, outputs.size(2)).squeeze()
        outputs = self.output_layer(outputs)
        return outputs
