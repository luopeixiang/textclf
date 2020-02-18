import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Classifier
from textclf.config import DPCNNClassifierConfig


class DPCNNClassifier(Classifier):
    """
    Reference:
        Deep Pyramid Convolutional Neural Networks for Text Categorization
        https://github.com/Tencent/NeuralNLP-NeuralClassifier/blob/master/model/classification/dpcnn.py
    """

    def __init__(self, config: DPCNNClassifierConfig):
        super(DPCNNClassifier, self).__init__(config)

        self.num_kernels = config.num_kernels
        self.pooling_stride = config.pooling_stride
        self.kernel_size = config.kernel_size
        self.radius = int(self.kernel_size / 2)
        assert self.kernel_size % 2 == 1, "kernel should be odd!"
        self.convert_conv = torch.nn.Sequential(
            torch.nn.Conv1d(
                config.input_size, self.num_kernels,
                self.kernel_size, padding=self.radius)
        )

        self.convs = torch.nn.ModuleList([torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                self.num_kernels, self.num_kernels,
                self.kernel_size, padding=self.radius),
            torch.nn.ReLU(),
            torch.nn.Conv1d(
                self.num_kernels, self.num_kernels,
                self.kernel_size, padding=self.radius)
        ) for _ in range(config.blocks + 1)])

        self.dropout = nn.Dropout(p=config.dropout)
        self.output_layer = torch.nn.Linear(self.num_kernels, config.output_size)

    def forward(self, embedding, seq_lens):

        embedding = embedding.permute(0, 2, 1)
        conv_embedding = self.convert_conv(embedding)
        conv_features = self.convs[0](conv_embedding)
        conv_features = conv_embedding + conv_features

        for i in range(1, len(self.convs)):
            block_features = F.max_pool1d(
                conv_features, self.kernel_size, self.pooling_stride)
            conv_features = self.convs[i](block_features)
            conv_features = conv_features + block_features
        pool_out = F.max_pool1d(conv_features, conv_features.size(2)).squeeze()

        outputs = self.output_layer(self.dropout(pool_out))
        return outputs
