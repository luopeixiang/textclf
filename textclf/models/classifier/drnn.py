import torch
import torch.nn as nn
import torch.nn.functional as F

from textclf.config import DRNNClassifierConfig
from .base import Classifier
from .components import RNN


class DRNNClassifier(Classifier):
    def __init__(self, config: DRNNClassifierConfig):
        super(DRNNClassifier, self).__init__(config)

        self.window_size = config.window_size

        self.dropout = nn.Dropout(p=config.dropout)
        rnn_config = config.rnn_config
        rnn_config.input_size = config.input_size
        self.rnn = RNN(rnn_config)
        if rnn_config.bidirectional:
            hidden_size = rnn_config.hidden_size*2
        else:
            hidden_size = rnn_config.hidden_size
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        self.mlp = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, config.output_size)

    def forward(self, embedding, seq_lens):
        # padding zeros vectors
        batch_size, max_len, emb_size = embedding.size()
        padding = torch.zeros(batch_size, self.window_size-1, emb_size)
        padding = padding.to(embedding.device)
        embedding = torch.cat([padding, embedding], dim=1)

        outputs = []
        for i in range(max_len):
            input_t = embedding[:, i:i+self.window_size, :]
            # shape of output_t: [batch_size, hidden_size]
            _, output_t = self.rnn(input_t)
            outputs.append(output_t)
        outputs = torch.stack(outputs, dim=1)
        outputs = outputs.transpose(1, 2).contiguous()
        outputs = self.batch_norm(outputs).transpose(1, 2).contiguous()
        outputs = self.mlp(outputs)
        outputs = outputs.permute(0, 2, 1)
        outputs = F.max_pool1d(outputs, outputs.size(2)).squeeze()
        outputs = self.output_layer(outputs)

        return outputs
