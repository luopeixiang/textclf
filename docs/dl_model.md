本文件根据[../textclf/config/dl_model.py](../textclf/config/dl_model.py)自动生成

### DLModelConfig

深度学习模型可以看成两部分:
1.embedding层:比如word2vec, glove, bert, 或者随机初始化的向量
2.classifier层：比如TextCNN/全连接层/TextRNN等等

DLModelConfig有以下属性：

 | Attribute name   | Type                 | Default                      | Description              |
|------------------|----------------------|------------------------------|--------------------------|
| embedding_layer  | EmbeddingLayerConfig | StaticEmbeddingLayerConfig() |                          |
| classifier       | ClassifierConfig     | CNNClassifierConfig()        | default model is textcnn |

