# 获取数据加载器的函数
# 核心是实现对齐的Batch，否则Transformer就黯然失色
# 将输入的一个batch的dialog转换成id序列，填充pad
import numpy as np
import torch
from Map import Batch_Pad_Dialogs


def DataLoader(dataset, vocab, batch_size):
    one_batch = []

    for input_seq, target_seq in dataset:
        one_batch.append([input_seq, target_seq])

        if len(one_batch) == batch_size:  # 一个 batch 的句子数量达到 batch_size 就 yield 出去并清空
            input_batch = []
            output_batch = []
            for dialog in one_batch:
                input_batch.append(dialog[0])  # batch 中每一个 dialog 的 input_sequence
                output_batch.append(dialog[1])  # batch 中每一个 dialog 的 target_sequence

            # 将一个batch中的输入语句和目标语句：word转化为id；按照一个batch中最长语句PAD
            input_padded_tensor = Batch_Pad_Dialogs(vocab, input_batch)
            target_padded_tensor = Batch_Pad_Dialogs(vocab, output_batch)

            # 清空临时区
            one_batch.clear()

            # 这是针对 Input_Batch 中最长句子 和 （对应的）Target_Batch 中最长句子 的长度不同的应对方法
            # 这种情况出现在：多轮对话中的 第一句话 或 最后一句话 是最长的。因为第一句只作为input不作为target，最后一句只作为target不作为input，所以这两种句子不会同时存在于input_batch和target_batch中
            if input_padded_tensor.shape[1] != target_padded_tensor.shape[1]:

                Pad_padding = torch.nn.ZeroPad2d((0, np.abs(input_padded_tensor.shape[1] - target_padded_tensor.shape[1]), 0, 0))
                # Pad_EOS = torch.nn.ConstantPad1d((0, 1), vocab.word2index['EOS'])

                if input_padded_tensor.shape[1] < target_padded_tensor.shape[1]:
                    # pop_EOS = input_padded_tensor[:, :-1]
                    pad_2_Equal = Pad_padding(input_padded_tensor)
                    # pad_EOS = Pad_EOS(pad_2_Equal)

                    # yield [pad_EOS, target_padded_tensor]
                    yield [pad_2_Equal, target_padded_tensor]

                else:
                    # pop_EOS = target_padded_tensor[:, :-1]
                    pad_2_Equal = Pad_padding(target_padded_tensor)
                    # pad_EOS = Pad_EOS(pad_2_Equal)

                    # yield [input_padded_tensor, pad_EOS]
                    yield [input_padded_tensor, pad_2_Equal]

            else:
                # 实例化出一个对象后，返回会从上一次暂停的地方继续，而不是重新开始。所以语料不会重复。
                yield [input_padded_tensor, target_padded_tensor]

        else:  # 最后一个Batch不够batch_size大小时，还没进入if循环就遍历完了，所以不会yield一个不足的结果出来
            pass  # keep add dialogs into a batch
