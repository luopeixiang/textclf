本文件根据[../textclf/config/embedding_layer.py](../textclf/config/embedding_layer.py)自动生成

### EmbeddingLayerConfig

 无可设置的属性



### StaticEmbeddingLayerConfig

静态词向量：word2vec, glove 或者随机初始化的向量

StaticEmbeddingLayerConfig继承EmbeddingLayerConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type                 | Default                               | Description                                                                                                                                                                               |
|------------------|----------------------|---------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dim              | int                  | 300                                   |                                                                                                                                                                                           |
| method           | str                  | "random"  # choice: random/pretrained |                                                                                                                                                                                           |
| pretrained_path  | Optional[str]        | None                                  | Specify pretrained model path, valid when methed is pretrainedThe pre-trained vector files are in text format.Each line contains a word and its vector. Each value is separated by space. |
| dictionary       | Optional[Dictionary] | None                                  | 字典，无效指定，运行时确定                                                                                                                                                                |



### BertEmbeddingLayerConfig



BertEmbeddingLayerConfig继承EmbeddingLayerConfig的所有属性，同时它还有以下属性：

 | Attribute name   | Type   | Default         | Description                                                       |
|------------------|--------|-----------------|-------------------------------------------------------------------|
| dim              | int    | 768             | the hidden size of hidden 如果是base则为768,如果是large，则为1024 |
| model_dir        | str    | "bert_pretrain" |                                                                   |

