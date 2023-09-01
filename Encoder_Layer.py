import torch.nn as nn
from Attention import Multi_Head_Attention
from Forward import Feed_Forward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_middle):
        super(EncoderLayer, self).__init__()

        self.MHA = Multi_Head_Attention(d_model, num_heads)
        self.FFN = Feed_Forward(d_middle, d_model).cuda()

        self.LayerNorm = nn.LayerNorm(normalized_shape=d_model).cuda()  # Multi-Head Attention的输出的最后一维是d_model（别忘了最后有一个全连接层）

        self.drop = nn.Dropout(0.1).cuda()

    def forward(self, x, Padding_Mask):
        mha_representation, _ = self.MHA(x, x, x, Mask=Padding_Mask)  # 编码器的第一个注意力是自注意力，所以只需要一个参数 x
        mha_representation = self.drop(mha_representation)
        Add_Norm_1 = self.LayerNorm(x + mha_representation)  # x:[..., seq_len_Q, depth_Q]    mha_representation:[..., seq_len_Q, d_model]    所以这里会做广播

        FF = self.drop(self.FFN(Add_Norm_1))
        Add_Norm_2 = self.LayerNorm(Add_Norm_1 + FF)

        return Add_Norm_2
