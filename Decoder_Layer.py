import torch.nn as nn
from Attention import Multi_Head_Attention
from Forward import Feed_Forward


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_middle):
        super(DecoderLayer, self).__init__()

        self.MHA = Multi_Head_Attention(d_model, num_heads)

        self.FFN = Feed_Forward(d_middle, d_model)

        self.Layer_Norm_1_2 = nn.LayerNorm(normalized_shape=d_model).cuda()  # 第一、二个接在注意力后面
        self.Layer_Norm_3 = nn.LayerNorm(normalized_shape=d_model).cuda()  # 第三个接在最后的FFN后面

        self.drop = nn.Dropout(0.1)

    def forward(self, x, Encoder_Output, Look_ahead_Mask, Padding_Mask):
        mha_representation_1, attention_weights_1 = self.MHA(x, x, x, Look_ahead_Mask)  # 注意：第一个是'自'注意力，且是'前瞻'掩码，无需边补掩码
        mha_representation_1 = self.drop(mha_representation_1)
        out_1 = self.Layer_Norm_1_2(x + mha_representation_1)

        mha_representation_2, attention_weights_2 = self.MHA(out_1, Encoder_Output, Encoder_Output, Padding_Mask)
        mha_representation_2 = self.drop(mha_representation_2)
        out_2 = self.Layer_Norm_1_2(out_1 + mha_representation_2)

        FF = self.drop(self.FFN(out_2))
        out_3 = self.Layer_Norm_3(out_2 + FF)

        return out_3, attention_weights_1, attention_weights_2
