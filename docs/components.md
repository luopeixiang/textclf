本文件根据[../textclf/config/components.py](../textclf/config/components.py)自动生成

### RNNConfig



RNNConfig有以下属性：

 | Attribute name   | Type   | Default   | Description                                                                     |
|------------------|--------|-----------|---------------------------------------------------------------------------------|
| rnn_type         | str    | "LSTM"    | specity the type of rnn, choices: RNN, GRU or LSTM                              |
| hidden_size      | int    | 256       | The number of features in the hidden state h                                    |
| num_layers       | int    | 2         | Number of recurrent layers. E.g., setting num_layers=2 would mean stacking      |
| bias             | bool   | True      | If False, then the layer does not use bias weights b_ih and b_hh. Default: True |
| dropout          | float  | 0.        | If non-zero, introduces a Dropout layer on the outputs of each LSTM layer       |
| bidirectional    | bool   | True      | If True, becomes a bidirectional LSTM. Default: False                           |

