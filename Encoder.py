import torch
import torch.nn as nn
from Position import Position
from Encoder_Layer import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_middle, input_vocab_size, maximum_position):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.pos_encoding = Position(maximum_position, d_model)

        self.encoder_layers = [EncoderLayer(d_model, num_heads, d_middle) for _ in range(num_layers)]

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, Padding_Mask):
        seq_len = x.shape[1]

        x = self.embedding(x.to('cuda:0', dtype=torch.int))
        x = x * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))  # 每个元素
        x = x + self.pos_encoding[:, :seq_len, :]  # maximum_position 是句子最大长度，但每次输入的句子长度都不同：[N, 每一句的Length, d_model]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x, Padding_Mask)

        return x
