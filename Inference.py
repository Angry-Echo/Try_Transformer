from Map import Sentence_To_Id
import torch
from Train import Create_Mask


def Infer(vocab, transformer, input_sequence, Max_Length):
    # 输入语句 --> id  在网络内部再Embedding
    input_sentence = Sentence_To_Id(vocab, input_sequence)
    encoder_input = torch.tensor(input_sentence).unsqueeze(0).cuda()  # 扩 batch_size == 1

    # 解码器的第一个输入是开始符：【SOS]
    decoder_input = torch.tensor([vocab.sos_token]).unsqueeze(0)

    # 自回归
    for i in range(Max_Length):
        enc_padding_mask, combined_mask, dec_padding_mask = Create_Mask(encoder_input, decoder_input)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input, decoder_input, enc_padding_mask, combined_mask, dec_padding_mask)

        # 从 seq_len 维度选择最后一个词
        # seq_len是什么？请回到Attention.py仔细看注释，是seq_len_Q，也就是Query的长度
        # 而在Decoder中，Query是Decoder的输'入'，Key是Encoder的输'出'，所以predictions的shape是[N, len(dec_inp), vocab_size]

        # 所以当                                dec_inp = [SOS] ->  [SOS 我]  -> [SOS 我 爱]   -> [SOS 我 爱 你]
        # 每次取dec_out的'最后一个字'回传给dec_inp(自回归):      ↓  回↗传  ↓  ↓   回↗传 ↓  ↓  ↓   回↗传  ↓  ↓  ↓  ↓
        # 对应的                                dec_out =  [我] ->   [我 爱]   -> [我 爱 你]    -> [我 爱 你 EOS]  -> 检测到EOS，结束

        last_word = predictions[:, -1:, :]  # (batch_size=1, 1, vocab_size)

        # 预测概率最大的那个词的索引
        word_index = torch.argmax(last_word, dim=-1)

        # 这里如果有疑问那可能是对训练时使用的交叉熵损失函数没有透彻理解，记住预测的策略与训练的目标是一样的
        # 如果预测概率最大的那个词的索引是词表是EOS的id，则返回结果
        if word_index == vocab.word2index['EOS']:
            word_index = last_word.squeeze().sort(0, True)[1][1]
            word_index = word_index.unsqueeze(0).unsqueeze(0)
            # return decoder_input.squeeze(0), attention_weights  # 返回dec_inp是因为dec_inp和dec_out是一样的，只是比dec_out多一个[SOS]少一个[EOS]
        # else:
            # 自回归
        decoder_input = torch.concat([decoder_input.to('cuda:0'), word_index], dim=-1)

    return decoder_input.squeeze(0), attention_weights


if __name__ == '__main__':
    from Dataset import corpus_set
    from Vocab import Vocab
    from Transformer import Transformer

    input_sequence = '你 在 干 什么 呢 ？'

    PAD_token = 0  # 补足句长的pad占位符的index
    SOS_token = 1  # 代表一句话开头的占位符的index
    EOS_token = 2  # 代表一句话结尾的占位符的index
    UNK_token = 3  # 代表不在词典中的字符

    MIN_COUNT = 2  # trim方法的修剪阈值

    LCCC_train_set = corpus_set(path='./dataset/LCCC-base-split/LCCC-base_train.json')
    voc = Vocab(name="LCCC-train-corpus", pad_token=PAD_token, sos_token=SOS_token, eos_token=EOS_token, unk_token=UNK_token)
    voc.load_data(corpus_set=LCCC_train_set)
    voc.trim(MIN_COUNT)

    num_layers = 4
    d_model = 128
    d_middle = 512
    num_heads = 8
    LR = 0.0001

    # 模型初始化
    transformer = Transformer(num_layers, d_model, num_heads, d_middle,
                              voc.num_words, voc.num_words,  # 多轮对话拆分为单轮让很多target同时也是input，且 train_set >> valid_set(强迫症同学自行分开input和output喔)
                              LCCC_train_set.max_length(), LCCC_train_set.max_length(),
                              # 所有对话中最长句子的长度(在Enc中根据每句话的长度对Position Matrix做了截断)，且我笃定train_set中的最长句子比valid_set中的长
                              )
    transformer = transformer.cuda()

    transformer.load_state_dict(torch.load(r'E:\AngryEcho\Try_Transformer\1_知乎从0开始系列\checkpoints\02.pth'))
    result, attention_weights = Infer(voc, transformer, input_sequence, Max_Length=30)

    print(result)

    print('Input: {}'.format(input_sequence))
    for i in result.cpu().numpy():
        # if i < tokenizer_en.vocab_size
        print(voc.index2word[i], end=' ')
