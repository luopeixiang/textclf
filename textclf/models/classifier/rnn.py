import torch.nn as nn

from textclf.config import RNNClassifierConfig
from .base import Classifier
from .components import RNN, AttentionLayer


class RNNClassifier(Classifier):
    def __init__(self, config: RNNClassifierConfig):
        super(RNNClassifier, self).__init__(config)

        self.use_attention = config.use_attention
        rnn_config = config.rnn_config
        rnn_config.input_size = config.input_size
        self.rnn = RNN(rnn_config)

        if rnn_config.bidirectional:
            hidden_size = rnn_config.hidden_size*2
        else:
            hidden_size = rnn_config.hidden_size
        self.dropout = nn.Dropout(p=config.dropout)
        self.output_layer = nn.Linear(hidden_size, config.output_size)

        if self.use_attention:
            self.attention_layer = AttentionLayer(hidden_size, 300)

    def forward(self, embedding, seq_lens):
        outputs, last_hidden = self.rnn(embedding, seq_lens)
        if self.use_attention:  # attention mechanism
            context = self.attention_layer(outputs, seq_lens)
        else:
            context = last_hidden

        logits = self.output_layer(self.dropout(context))
        return logits
