import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

from textclf.config.components import RNNConfig


class RNN(torch.nn.Module):
    """
    rnn layer
    """
    supported_types = ["RNN", "GRU", "LSTM"]

    def __init__(self, config: RNNConfig):
        super(RNN, self).__init__()
        self.rnn_type = config.rnn_type
        self.num_layers = config.num_layers
        self.bidirectional = config.bidirectional

        if config.rnn_type not in self.supported_types:
            types_str = " ".join(self.supported_types)
            msg = f"Unsupported rnn init type: {config.rnn_type}."
            msg += f" Supported rnn type is: {types_str}"
            raise TypeError(msg)

        self.rnn = getattr(torch.nn, config.rnn_type)(
            config.input_size,
            config.hidden_size,
            num_layers=config.num_layers,
            bias=config.bias,
            dropout=config.dropout,
            bidirectional=config.bidirectional,
            batch_first=True
        )

    def forward(self, inputs, seq_lengths=None, init_state=None):
        if seq_lengths is not None:
            seq_lengths = seq_lengths.int()
            sorted_seq_lengths, indices = torch.sort(seq_lengths, descending=True)
            sorted_inputs = inputs[indices]
            packed_inputs = pack_padded_sequence(
                sorted_inputs,
                sorted_seq_lengths,
                batch_first=True
            )
            outputs, state = self.rnn(packed_inputs, init_state)
        else:
            outputs, state = self.rnn(inputs, init_state)

        if self.rnn_type == "LSTM":
            state = state[0]

        if self.bidirectional:  # concatenate bidirectional hidden state
            last_layers_hn = state[2 * (self.num_layers - 1):]
            last_layers_hn = torch.cat((last_layers_hn[0], last_layers_hn[1]), 1)
        else:
            last_layers_hn = state[self.num_layers - 1:]
            last_layers_hn = last_layers_hn[0]

        if seq_lengths is not None:
            # re index
            _, revert_indices = torch.sort(indices, descending=False)
            last_layers_hn = last_layers_hn[revert_indices]
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)
            outputs = outputs[revert_indices]
        return outputs, last_layers_hn


class AttentionLayer(nn.Module):
    def __init__(self, input_dim, attention_dim, dropout=0):
        super(AttentionLayer, self).__init__()
        self.attention_matrix = nn.Linear(input_dim, attention_dim)
        self.attention_vector = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, inputs, seq_lens):
        u = torch.tanh(self.attention_matrix(inputs))

        attn_logits = self.attention_vector(u).squeeze()
        for i, seq_len in enumerate(seq_lens):
            attn_logits[i][seq_len.item():] = -1e9

        alpha = F.softmax(attn_logits, 1).unsqueeze(1)
        context = torch.matmul(alpha, inputs).squeeze()
        return context
