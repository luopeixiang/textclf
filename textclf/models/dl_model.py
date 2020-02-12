import torch.nn as nn

from textclf.utils.create import create_instance
from textclf.config import DLModelConfig


class DLModel(nn.Module):
    def __init__(self, config: DLModelConfig):
        super(DLModel, self).__init__()
        self.config = config

        self.embedding_layer = create_instance(self.config.embedding_layer)
        self.config.classifier.input_size = self.config.embedding_layer.dim
        self.classifier = create_instance(self.config.classifier)

    def forward(self, ids, lens):
        """
        :param ids: `torch.LongTensor` shape of [batch_size, max_len]
        :param lens: `torch.LongTensor`, shape of [batch_size]

        Return: logits: `torch.FloatTensor`, shape of [batch_size, label_size]
        """
        embedding = self.embedding_layer(ids)
        logits = self.classifier(embedding, lens)
        return logits
