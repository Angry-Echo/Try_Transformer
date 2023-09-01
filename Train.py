import torch
from Mask import Padding_Mask, Look_ahead_Mask
from Mask import Create_Mask


def Train(input_seq, target_seq, transformer, loss_function, loss_obj, optimizer):
    """
    原目标句子：SOS A lion in the jungle is sleeping EOS
    因为 自回归 + Teacher Forcing，Decoder的 输入 和 输出 要偏移一个单词
    所以 Dec的输入是 SOS   A   lion   in    the    jungle   is      sleeping
             输出是  A   lion  in    the  jungle    is   sleeping     EOS
    """
    Dec_inp = input_seq[:, :-1]  # 去掉 输入 的每一句话（每一行）的最后一个'EOS'，保留第一个'SOS'
    Dec_tgt = target_seq[:, 1:]  # 去掉 目标 的每一句话（每一行）的第一个符号'SOS'，保留最后一个'EOS'

    # Enc的自注意力的Padding 和 Dec中Intro的Padding 都根据Enc的Input_Seq创建，所以这两个共用一个参数：input_seq
    # Dec的 Look_ahead 和 Padding 则根据 Dec的Input_Seq创建，所以需要的参数是：Dec_inp
    # 这里的 输入语句（包括编码器输入input_seq和解码器输入Dec_inp）的padding的id是0，所以可以根据0轻易的制作Mask位置矩阵
    # 等到进入网络后，即使是padding的地方，加上了position，就不再是0了，但是原本是0的地方已经通过Mask记下了
    # 具体可以看Encoder.py中forward函数从position中截取的长度seq_len = x.shape[1]，是已经padding过的长度
    # 再在Decoder中的
    enc_padding_mask, combined_mask, dec_padding_mask = Create_Mask(input_seq, Dec_inp)

    # transformer是对象，这里调用的是forward方法
    predictions, _ = transformer(input_seq, Dec_inp, enc_padding_mask, combined_mask, dec_padding_mask)
    loss = loss_function(predictions.reshape(predictions.shape[0] * predictions.shape[1], -1), Dec_tgt.reshape(Dec_tgt.shape[0] * Dec_tgt.shape[1]), loss_obj)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def Valid(input_seq, target_seq, transformer, loss_function, loss_obj):
    Dec_inp = input_seq[:, :-1]  # 去掉 输入 的每一句话（每一行）的最后一个'EOS'，保留第一个'SOS'
    Dec_tgt = target_seq[:, 1:]  # 去掉 目标 的每一句话（每一行）的第一个符号'SOS'，保留最后一个'EOS'

    # Enc的自注意力的Padding 和 Dec中Intro的Padding 都根据Enc的Input_Seq创建，所以这两个共用一个参数：input_seq
    # Dec的 Look_ahead 和 Padding 则根据 Dec的Input_Seq创建，所以需要的参数是：Dec_inp
    enc_padding_mask, combined_mask, dec_padding_mask = Create_Mask(input_seq, Dec_inp)

    # transformer是对象，这里调用的是forward方法
    predictions, _ = transformer(input_seq, Dec_inp, enc_padding_mask, combined_mask, dec_padding_mask)
    loss = loss_function(predictions.reshape(predictions.shape[0] * predictions.shape[1], -1),
                         Dec_tgt.reshape(Dec_tgt.shape[0] * Dec_tgt.shape[1], -1).squeeze(), loss_obj)

    return loss.cpu().numpy()


class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after certain epochs.
    """

    def __init__(self, patience=7, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss

        elif val_loss - self.best_loss > self.min_delta:
            self.counter += 1
            print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.stop = True

        else:
            self.best_loss = val_loss
            self.counter = 0
