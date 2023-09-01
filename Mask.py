"""
两种 Mask 的实现
第一种：Padding Mask
第二种：Look-ahead Mask
"""
import numpy as np
import torch


def Padding_Mask(seq):
    """
    前提：补足语（padding）在张量中的元素为 0
    生成一个与'输入'的维度大小完全相同的标记矩阵，其中'输入'中：自然语言对应的位置为False；padding的位置填True
    注意：所谓 Mask，并不是在'输入'上做遮挡，而是先记下张量中无意义的元素的位置（并没有对seq做任何修改），最后在计算SoftMax时给这些位置打上-∞的分数，详见 Attention.py
    :param seq: 有非正常元素的语言的张量   维度：【 N, ..., Max_Seq_len 】 （最后一维是一句话的最大单词数（含补足语））
    :return: Tensor[N，1，1，seq_len]  为了能够触发”广播“来匹配 attention_weight
    """

    signal_mat = torch.eq(seq, 0)  # 参数二：0 代表相同size的全0张量（广播）   匹配相同（0）为True
    mask_mat = signal_mat.to(device='cuda:0', dtype=torch.float32)  # 转为0-1矩阵

    return mask_mat[:, np.newaxis, np.newaxis, :].cuda()


def Look_ahead_Mask(size):
    """
    预测当前的单词，只允许看到当前位置'之前'的所有单词（训练时的TeacherForcing）
    前瞻遮挡 不用知晓'被遮语句的嵌入向量'中的具体元素是什么，只需要知道它的size即可
    因为解码器是'逐字'生成，所以默认做下三角矩阵即可（一字符位移）
    PS: 建议修改为：输入tgt，函数体内部获取其size
    :param size: 当前预测的target（对应当前输入的source）的大小
    :return: 与target相同size的下三角标记矩阵
    """

    allow_mat = torch.tril(torch.ones((size, size)))  # 下三角的1 代表允许的输入
    mask_mat = 1 - allow_mat  # 需要遮掩的位置是剩余的0的位置，故取反

    return mask_mat.cuda()


def Create_Mask(input_seq, target_seq):  # shape: input_seq = target_seq = [N, Seq_Len]
    # 编码器——自注意力——补语遮挡
    enc_self_padding = Padding_Mask(input_seq)

    # 解码器——交叉注意力——补语遮挡
    # 用于遮挡 Enc -> Dec 的传递
    # 这里遮挡的机理仍是根据 input_seq 中的 padding 的位置决定哪些位置的概率得分没有意义
    # 所以 传入给函数的矩阵仍是 input_seq， 即使 Enc的 output的维度是[N, Seq_Len, d_model]
    dec_intro_padding = Padding_Mask(input_seq)

    # 解码器——自注意力——（补语遮挡 + 前瞻遮挡）
    dec_self_ahead = Look_ahead_Mask(target_seq.shape[1])  # ahead 只需要知道当前句子的长度即可（所以看shape中的SeqLen维度就行），它是个正方形
    dec_self_padding = Padding_Mask(target_seq)
    combined_mask = torch.maximum(dec_self_ahead, dec_self_padding)

    return enc_self_padding, combined_mask, dec_intro_padding
