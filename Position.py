"""
Transformer的并行训练 :矩阵输入，每行并行（蕴含Teacher Forcing）
[ <sos>
  <sos>, I
  <sos>, I, am
  <sos>, I, am, IronMan ]
同时输入，同时计算，无时间上的先后关系
所以要’加‘上 Position Encoding，使得文字转换成的张量中‘含有’‘语序’信息
"""

import numpy as np
import torch


def Position(position: int, d_model: int) -> torch.Tensor:
    """
    先分别获取 sin 和 cos 下的完整矩阵
    再组合两个矩阵的奇偶列
    :param position: 行数（最大位置长度？）
    :param d_model: 列数（词向量的维度数？）
    :return:位置编码词表（含有位置信息的矩阵）
    """
    # 矩阵，展开！
    # 扩张维度是为了相除时触发广播机制
    A = np.arange(position)[:, np.newaxis]  # 行×列：position×1
    B = np.arange(d_model)[np.newaxis, :]  # 行×列：1×d_model

    '''
    (1) 融合位置信息是直接将'位置编码'与 Word Embedding 进行element-wise的相加
    (2) 所以事先生成一个与 Word Embedding 相同大小的位置信息矩阵，position：矩阵的行数，d_model：矩阵的列数
    (3) 位置信息矩阵中 每个位置的值 根据公式计算，其中 pos 和 i 分别指代矩阵中每个位置的下标
    (4) pos代表行号，i代表列号，这已经通过'np.arange'生成在A、B矩阵中，以矩阵为单位一次完成运算
    '''
    # 相同的三角函数内的函数
    # 广播机制：A原本只有一列，复制为与B同列数；B原本只有一行，复制为与A同行数
    # 广播为相同维度后，对应位置的两个元素之间进行计算，结果是根据位置确定的位置信息表
    same = A / np.power(10000, (2 * B) / d_model)
    pos_mat = np.zeros((position, d_model))
    pos_mat[:, 0::2] = np.sin(same[:, 0::2])  # 偶数列用sin，奇数列用cos
    pos_mat[:, 1::2] = np.cos(same[:, 0::2])

    # 在最外层增加一个维度：N，可以触发广播（自动扩列），为一个batch内的所有句子加上position
    pos_mat = pos_mat[np.newaxis, ...]

    # 转换为张量，独立的内存
    pos_mat = torch.tensor(pos_mat, dtype=torch.float32, device=torch.device('cuda:0'), requires_grad=True)

    return pos_mat
