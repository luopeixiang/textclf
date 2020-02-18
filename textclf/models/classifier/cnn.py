import torch

from .base import Classifier
from textclf.config import CNNClassifierConfig


class CNNClassifier(Classifier):
    def __init__(self, config: CNNClassifierConfig):
        super(CNNClassifier, self).__init__(config)
        self.kernel_sizes = self.config.kernel_sizes
        self.convs = torch.nn.ModuleList()
        for kernel_size in self.kernel_sizes:
            self.convs.append(
                torch.nn.Conv1d(
                    self.config.input_size,
                    self.config.num_kernels,
                    kernel_size,
                    padding=kernel_size - 1
                )
            )

        self.top_k = self.config.top_k_max_pooling
        hidden_size = len(self.config.kernel_sizes) * \
            self.config.num_kernels * self.top_k
        self.linear = torch.nn.Linear(hidden_size, self.config.output_size)
        self.dropout = torch.nn.Dropout(p=self.config.hidden_layer_dropout)

    def forward(self, embedding, seq_lens):
        # 一维卷积是在最后维度上扫的，所以需要transpose
        embedding = embedding.transpose(1, 2)  # [batch_size, embedding_dim, max_len]
        pooled_outputs = []
        for i, conv in enumerate(self.convs):
            convolution = torch.nn.functional.relu(conv(embedding))
            pooled = torch.topk(convolution, self.top_k)[0].view(convolution.size(0), -1)
            pooled_outputs.append(pooled)

        text_embedding = torch.cat(pooled_outputs, 1)
        return self.linear(self.dropout(text_embedding))
