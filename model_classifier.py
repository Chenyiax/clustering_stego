"""
文件名: classifier_model.py
作者: 徐辰屹
日期: 2024年5月21日

说明: 载体模型文件
"""
import math

import torch
from torch import nn
from torchvision import models

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_dim=256, output_dim=2, num_layers=2):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True,
                            dropout=0.5)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        output = self.fc(hidden)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    def __init__(self, vocab_size, vocab_len, dim_model=512, nums=2, num_layers=6):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, dim_model)
        self.positional_encoding = PositionalEncoding(dim_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_model, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(vocab_len * dim_model, nums)
        self.transformer_encoder.layers[0].self_attn.out_proj.reset_parameters()
        self.transformer_encoder.layers[0].linear1.reset_parameters()

    def forward(self, x):
        out = self.embedding(x)
        out = self.positional_encoding(out)
        out = out.permute(1, 0, 2)  # Change to (seq_len, batch_size, dim_model) for Transformer
        out = self.transformer_encoder(out)
        out = out.permute(1, 0, 2)  # Change back to (batch_size, seq_len, dim_model)
        out = out.contiguous().view(out.size(0), -1)  # Flatten
        out = self.output_layer(out)
        return out
