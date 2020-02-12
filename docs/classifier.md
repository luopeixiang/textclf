本文件根据[../textclf/config/classifier.py](../textclf/config/classifier.py)自动生成

### ClassifierConfig



ClassifierConfig有以下属性：

 | Attribute name   | Type          | Default   | Description                                      |
|-------------------------------------------------------------------------------------------------|
| input_size       | Optional[int] | None      | input_size是embedding layer层的维度              |
| output_size      | Optional[int] | None      | output_size是输出空间的大小,应该与标签数量相对应 |



### CNNClassifierConfig



CNNClassifierConfig继承ClassifierConfig的所有属性，同时它还有以下属性：

 | Attribute name       | Type      | Default                 | Description   |
|----------------------------------------------------------------------------|
| kernel_sizes         | List[int] | [2, 3, 4]               |               |
| num_kernels          | int       | 256                     |               |
| top_k_max_pooling    | int       | 1  # max top-k pooling. |               |
| hidden_layer_dropout | float     | 0.5                     |               |



### LinearClassifierConfig



LinearClassifierConfig继承ClassifierConfig的所有属性，同时它还有以下属性：

 | Attribute name    | Type                  | Default   | Description                                                                                                                        |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| pool_method       | str                   | "first"   | first or meanif first, use first token's embedidng as Linear layer inputif mean, average sequences embedding as linear layer input |
| embedding_dropout | float                 | 0.1       | dropout probability of the embedding layer                                                                                         |
| hidden_units      | Optional[List[int]]   | None      | 隐藏层设置                                                                                                                         |
| activations       | Optional[List[str]]   | None      |                                                                                                                                    |
| hidden_dropouts   | Optional[List[float]] | None      |                                                                                                                                    |



### RNNClassifierConfig



RNNClassifierConfig继承ClassifierConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type   | Default   | Description   |
|-------------------------------------------------------|



### RCNNClassifierConfig



RCNNClassifierConfig继承ClassifierConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type   | Default   | Description   |
|-------------------------------------------------------|



### RNNAttnClassifierConfig



RNNAttnClassifierConfig继承ClassifierConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type   | Default   | Description   |
|-------------------------------------------------------|

