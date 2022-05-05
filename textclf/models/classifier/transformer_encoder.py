import torch
import torch.nn as nn

from .base import Classifier
from textclf.config import TransformerClassifierConfig
from textclf.utils.dl_data import seqLens_to_mask


class TransformerClassifier(Classifier):
    def __init__(self, config: TransformerClassifierConfig):
        super(TransformerClassifier, self).__init__(config)
        encoder_layer = nn.TransformerEncoderLayer(
            config.d_model, config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=config.batch_first,
            norm_first=config.norm_first
        )
        encoder_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.encoder = nn.TransformerEncoder(
            encoder_layer, config.num_encoder_layers, encoder_norm)

        self.dropout = nn.Dropout(p=config.dropout)
        self.output_layer = nn.Linear(config.d_model, config.output_size)

    def forward(self, embedding, seq_lens):
        mask = seqLens_to_mask(seq_lens)
        encoder_outputs = self.encoder(embedding, src_key_padding_mask=mask)  # [B, Max_len, EmbSize]
        hidden = torch.mean(encoder_outputs, 1)
        return self.output_layer(self.dropout(hidden))
