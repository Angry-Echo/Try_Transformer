import torch
import torch.nn as nn
from Position import Position
from Decoder_Layer import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_middle, target_vocab_size, maximum_position_encoding):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = nn.Embedding(target_vocab_size, d_model)
        self.pos_encoding = Position(maximum_position_encoding, d_model)

        self.decoder_layers = [DecoderLayer(d_model, num_heads, d_middle) for _ in range(num_layers)]

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, Encoder_Output, Look_ahead_Mask, Padding_Mask):
        seq_len = x.shape[1]
        attention_weights = {}

        x = self.embedding(x.to('cuda:0', dtype=torch.long))  # [N, target_seqlen, d_model]
        x = x * torch.sqrt(torch.tensor(self.d_model).to('cuda:0', dtype=torch.float32))
        x = x + self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, MHA_block_1, MHA_block_2 = self.decoder_layers[i](x, Encoder_Output, Look_ahead_Mask, Padding_Mask)  # (×N)中Encoder_Output没变
            attention_weights['Decoder_{}_block_1'.format(i + 1)] = MHA_block_1
            attention_weights['Decoder_{}_block_2'.format(i + 1)] = MHA_block_2

        return x, attention_weights  # [N, target_seqlen, d_model] 此处还不是输出，还没有映射到 target_vocab_size
