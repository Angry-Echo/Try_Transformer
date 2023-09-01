import torch
from itertools import zip_longest


def Sentence_To_Id(vocab, sentence):  # 将一句话转换成id序列(str->list)，为了让网络知道一句话什么时候结束，每一句结尾加上EOS
    seq_id = []

    for word in sentence.split():
        if word in vocab.word2index:
            seq_id.append(vocab.word2index[word])
        else:
            seq_id.append(vocab.word2index['UNK'])

    return [vocab.sos_token] + seq_id + [vocab.eos_token]


# 将一个batch中的input_dialog转化为有pad填充的tensor（batch_first）
def Batch_Pad_Dialogs(vocab, batch):
    # 多个句子（id）做成一个batch：[[1,2,3,...], [2,3,4,...], ...]
    batch_id_seqs = [Sentence_To_Id(vocab, sentence) for sentence in batch]

    '''
    print(max([len(id_seq) for id_seq in batch_id_seqs]))
    for seq in batch_id_seqs:
        if len(seq) == max([len(id_seq) for id_seq in batch_id_seqs]):
            print(f'Max_Seq:', end=' ')
            for id in seq:
                print(vocab.index2word[id], end=' ')
    '''

    # 以一个batch中最长的句子为准，所有短于这个句子的，填充0(PAD)
    unTransposed = list(zip_longest(*batch_id_seqs, fillvalue=vocab.pad_token))  # 一定要转化成list
    batched_padded_tensor = torch.tensor(unTransposed, dtype=torch.float32).t()  # 一定要转置

    return batched_padded_tensor
