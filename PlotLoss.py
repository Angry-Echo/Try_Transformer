from matplotlib import pyplot as plt


class Plot_loss:
    """Plot loss curve of train and valid"""

    def __init__(self, train_loss, valid_loss):
        self.train_loss, self.valid_loss = train_loss, valid_loss

    def matplot_loss(self):
        plt.plot(self.train_loss, label='train_loss')
        plt.plot(self.valid_loss, label='val_loss')
        plt.legend(loc='best')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title("测试集和验证集loss值对比图")
