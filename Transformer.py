import torch.nn as nn
from Encoder import Encoder
from Decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_middle, input_vocab_size, target_vocab_size, max_input_pos, max_target_pos):
        super(Transformer, self).__init__()

        self.Encoder = Encoder(num_layers, d_model, num_heads, d_middle, input_vocab_size, max_input_pos)
        self.Decoder = Decoder(num_layers, d_model, num_heads, d_middle, target_vocab_size, max_target_pos)

        self.final_linear = nn.Linear(d_model, target_vocab_size)

    def forward(self, input, target, Encoder_Padding_Mask, Look_ahead_Mask, Decoder_Padding_Mask):  # 注意看，作为整个Transformer，这里的输入不仅有input_sequence，还有target_sequence
        Encoder_Output = self.Encoder(input, Encoder_Padding_Mask)  # [N, input_seqLen, d_model]

        Decoder_Output, Attention_weights = self.Decoder(target, Encoder_Output, Look_ahead_Mask, Decoder_Padding_Mask)  # 解码器中的第一个MHA使用target和Look_ahead_Mask, 第二个MHA使用前一个的输出、Encoder_Output和Decoder_Padding_Mask

        Final_Output = self.final_linear(Decoder_Output)

        # 在 Pytorch中的CrossEntropyLoss中包含了SoftMax，所以此处无需再加论文中画出的SoftMax层
        return Final_Output, Attention_weights
