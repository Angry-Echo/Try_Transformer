"""
详解见工作记录：2023.3.21 与 幕布笔记
    此处比详解多一个维度 N，也就是【N，
Query-Key-Value彼此独立，是三维矩阵运算（N）
    矩阵乘法 或 转置 只针对最后的两个维度
"""
import torch
import torch.nn as nn


def scaled_dot_product_attention(Q, K, V, Mask):
    """
    注意事项：
    1. 这个 depth = d_model / num_heads，因为这个模块是放在'多头'中运行的子模块，其计算的是'多头'中的每个'单头'
    2. Q, K, V "前置"维度相同
    3. Q, K 满足 depth_Q = depth_K （转置）
    4. K, V 满足 seq_len_K = seq_len_V
    5. 虽然 Mask根据其类型（填充或前瞻）有不同的形状，但是 Mask必须能进行广播转换以便求和
    :param Q: [..., seq_len_Q, depth_Q]
    :param K: [..., seq_len_K, depth_K]
    :param V: [..., seq_len_V, depth_V]
    :param Mask: [..., seq_len_Q, seq_len_K]
    :return: 注意力表征  求和（权重*值）  [..., seq_len_Q, depth_V]
    """

    # 计算相似度
    Q_K = torch.matmul(Q, torch.transpose(K, -2, -1))  # [..., seqlen_Q, seqlen_K]

    # 取公式中的d_k  也就是Key的编码深度  进行缩放
    d_k = torch.tensor(K.shape[-1], dtype=torch.float32)  # 刚才的转置非in-place，别误会~
    scaled_similarity = Q_K / torch.sqrt(d_k)

    # 是否掩码
    if Mask is not None:
        scaled_similarity += (Mask * -1e9)  # 相同维度的Mask矩阵（元素的位置是关键，元素的值：不遮的是0，遮挡的是1），再与相似度分数矩阵相加，位置对齐，遮盖的相似度分数负无穷，不遮的不变

    # SoftMax 在最内层维度计算
    # 第一行：Query语句中的'第一个'单词 与 Key语句中的'每一个'单词 的相似度评分表
    # 第二行：Query语句中的'第二个'单词 与 ... （Query矩阵的第一行 * Key矩阵的每一列 = 相似度矩阵的第一行）
    # 维度：[..., seqlen_Q, seqlen_K]：Query的单词个数（行数）  Key的单词个数（列数）  Query与Key中每个单词的相似度都被计算了
    attention_weight = torch.softmax(scaled_similarity, dim=-1)

    # 乘上权重  权重 V 就是 Key自身（的词向量）  维度：[..., seq_len_V, depth_V] = [..., seq_len_K, depth_V]（向量深度可以不同）
    # 基于此，才可以进行矩阵相乘（m×s * s×n = m×n）=> [..., seq_len_Q, seqlen_K] * [..., seq_len_K, depth_V] = [..., seq_len_Q, depth_V]
    # 维度：[..., seq_len_Q, depth_V]
    representation = torch.matmul(attention_weight, V)

    # torch.matmul只对最后两个矩阵的最后两个维度做相乘

    return representation, attention_weight


class Multi_Head_Attention(nn.Module):
    def __init__(self, d_model, num_heads):
        """
        :param d_model: = num_heads * depth（必须能被 num_heads 整除）
        :param num_heads:
        """
        super(Multi_Head_Attention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % num_heads == 0

        self.depth = self.d_model // self.num_heads

    def split_heads(self, X, N):
        X = X.reshape(N, -1, self.num_heads, self.depth)  # 拆分最后一个维度：[N, seq_len, 'd_model'] -> [N, seq_len, 'nums_heads, depth']
        X = X.transpose(1, 2)  # 将维度还原为[N, num_heads, seq_len, depth]，始终要保持最后两个维度是[..., seq_len, depth]
        return X

    def __call__(self, Q, K, V, Mask):
        N = Q.shape[0]

        # 输入自注意力的 Query、Key 和 Value 都来自原始词嵌入的线性变换：创建三个独立的全连接层，随着训练优化
        Q = nn.Linear(Q.shape[-1], self.d_model).cuda()(Q)  # [N, seq_len_Q, d_model]
        K = nn.Linear(K.shape[-1], self.d_model).cuda()(K)
        V = nn.Linear(V.shape[-1], self.d_model).cuda()(V)

        sub_Q = self.split_heads(Q, N)  # [N, num_heads, seq_len_Q, depth]
        sub_K = self.split_heads(K, N)
        sub_V = self.split_heads(V, N)

        # 内部真的是按照多头的维度运算的？
        # [N, num_heads, seq_len_Q, seq_len_K], [N, num_heads, seq_len_Q, depth]
        representation, attention_weights = scaled_dot_product_attention(sub_Q, sub_K, sub_V, Mask)

        # 先将 num_heads 放到倒数第二维
        # 再将最后两个维度合并：[N, seq_len, 'nums_heads, depth'] -> [N, seq_len, 'd_model']
        concat_representation = representation.transpose(1, 2).reshape(N, -1, self.d_model)

        output_representation = nn.Linear(self.d_model, self.d_model).cuda()(concat_representation)

        return output_representation, attention_weights
