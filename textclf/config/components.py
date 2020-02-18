from .base import ConfigBase


class RNNConfig(ConfigBase):
    # specity the type of rnn, choices: RNN, GRU or LSTM
    rnn_type: str = "LSTM"

    # The number of features in the hidden state h
    hidden_size: int = 256

    # Number of recurrent layers. E.g., setting num_layers=2 would mean stacking
    num_layers: int = 2

    # If False, then the layer does not use bias weights b_ih and b_hh. Default: True
    bias: bool = True

    # If non-zero, introduces a Dropout layer on the outputs of each LSTM layer
    dropout: float = 0.

    # If True, becomes a bidirectional LSTM. Default: False
    bidirectional: bool = True
