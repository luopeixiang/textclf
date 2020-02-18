本文件根据[../textclf/config/classifier.py](../textclf/config/classifier.py)自动生成

### ClassifierConfig



ClassifierConfig有以下属性：

 | Attribute name   | Type          | Default   | Description                                      |
|------------------|---------------|-----------|--------------------------------------------------|
| input_size       | Optional[int] | None      | input_size是embedding layer层的维度              |
| output_size      | Optional[int] | None      | output_size是输出空间的大小,应该与标签数量相对应 |



### CNNClassifierConfig



CNNClassifierConfig继承ClassifierConfig的所有属性，同时它还有以下属性：

 | Attribute name       | Type      | Default                 | Description   |
|----------------------|-----------|-------------------------|---------------|
| kernel_sizes         | List[int] | [2, 3, 4]               |               |
| num_kernels          | int       | 256                     |               |
| top_k_max_pooling    | int       | 1  # max top-k pooling. |               |
| hidden_layer_dropout | float     | 0.5                     |               |



### LinearClassifierConfig



LinearClassifierConfig继承ClassifierConfig的所有属性，同时它还有以下属性：

 | Attribute name    | Type                  | Default   | Description                                                                                                                        |
|-------------------|-----------------------|-----------|------------------------------------------------------------------------------------------------------------------------------------|
| pool_method       | str                   | "first"   | first or meanif first, use first token's embedidng as Linear layer inputif mean, average sequences embedding as linear layer input |
| embedding_dropout | float                 | 0.1       | dropout probability of the embedding layer                                                                                         |
| hidden_units      | Optional[List[int]]   | None      | 隐藏层设置                                                                                                                         |
| activations       | Optional[List[str]]   | None      |                                                                                                                                    |
| hidden_dropouts   | Optional[List[float]] | None      |                                                                                                                                    |



### RNNClassifierConfig



RNNClassifierConfig继承ClassifierConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type      | Default     | Description                                              |
|------------------|-----------|-------------|----------------------------------------------------------|
| rnn_config       | RNNConfig | RNNConfig() | config for RNN layer                                     |
| use_attention    | bool      | True        | If True, use attention mechanism to caluate output state |
| dropout          | float     | 0.2         | dropout probability on context                           |



### RCNNClassifierConfig



RCNNClassifierConfig继承ClassifierConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type      | Default     | Description                                                                                                                               |
|------------------|-----------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------|
| rnn_config       | RNNConfig | RNNConfig() | config for RNN layer                                                                                                                      |
| semantic_units   | int       | 512         | size of  latent semantic vectorRefer to Equation 4 of the original paper"Recurrent Convolutional Neural Networks for Text Classification" |



### DRNNClassifierConfig



DRNNClassifierConfig继承ClassifierConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type      | Default     | Description                                                       |
|------------------|-----------|-------------|-------------------------------------------------------------------|
| rnn_config       | RNNConfig | RNNConfig() | config for RNN layer                                              |
| dropout          | float     | 0.2         | The dropout probability  applied  in drnninput and output layers, |
| window_size      | int       | 10          | The window size for rnn                                           |



### DPCNNClassifierConfig



DPCNNClassifierConfig继承ClassifierConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type   | Default   | Description                                 |
|------------------|--------|-----------|---------------------------------------------|
| kernel_size      | int    | 3         | kernel size.                                |
| pooling_stride   | int    | 2         | stride of pooling.                          |
| num_kernels      | int    | 16        | number of kernels.                          |
| blocks           | int    | 2         | number of blocks for DPCNN.                 |
| dropout          | float  | 0.2       | dropout probability on convolution features |

