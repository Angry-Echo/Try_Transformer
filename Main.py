import os
import gc
import torch
import torch.nn as nn
import torch.cuda as CUDA
import torch.optim as optimize
from Transformer import Transformer
from Vocab import Vocab
from DataLoader import DataLoader
from Dataset import corpus_set
from Train import Create_Mask, Train, Valid, EarlyStopping
from LossFunc import Loss
from PlotLoss import Plot_loss
import matplotlib.pyplot as plt

PAD_token = 0  # 补足句长的pad占位符的index
SOS_token = 1  # 代表一句话开头的占位符的index
EOS_token = 2  # 代表一句话结尾的占位符的index
UNK_token = 3  # 代表不在词典中的字符

BATCH_SIZE = 8  # 一个batch中的对话数量（样本数量）
MIN_COUNT = 3  # trim方法的修剪阈值

# 实例化数据集对象
LCCC_train_set = corpus_set(path='./dataset/LCCC-base-split/LCCC-base_train.json')
LCCC_valid_set = corpus_set(path='./dataset/LCCC-base-split/LCCC-base_valid.json')
print(f'当前的训练语料数据集的对话个数{len(LCCC_train_set)}')
print(f'当前的验证语料数据集的对话个数{len(LCCC_valid_set)}')

# 初始化词表类
voc = Vocab(name="LCCC-train-corpus", pad_token=PAD_token, sos_token=SOS_token, eos_token=EOS_token, unk_token=UNK_token)

# 建立词表
voc.load_data(corpus_set=LCCC_train_set)

# 词典剪枝（不是剪数据集！）
voc.trim(MIN_COUNT)

# 数据处理测试
'''
batch_item_names = ["batched_padded_input_tensor", "batched_padded_target_tensor"]
for batch in loader:
    for name, item in zip(batch_item_names, batch):
        print(f"{name} :\n {item}")
        for id_seq_tensor in item:
            for id_tensor in id_seq_tensor:
                id = int(id_tensor)
                print(voc.index2word[id])
    break
'''

# 一个灯泡
'''
既然我现在将语料按’词‘分，即使预测是逐步，也应该是以’词‘为最小单位生成语句吧，毕竟你生成的id在词典中转换出的是词（当然也包括一些单字）。
所以送回其实也是按词，因为送回的是id，id代表的是词
'''

# 硬件配置
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gc.collect()
CUDA.empty_cache()

# 确定随机性
if torch.cuda.is_available():
    print('GPU is available!')

    torch.manual_seed(42)
    CUDA.manual_seed(42)

    os.environ['PYTHONHASHSEED'] = str(42)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True  # for Convolution Operation
    torch.backends.cudnn.enabled = False

# 无聊的参数定义(ˉ▽ˉ；)...
loss_obj = nn.CrossEntropyLoss(reduction='none')
num_layers = 6
d_model = 256
d_middle = 512
num_heads = 8
LR = 0.0003

# 重新初始化——count归为0
early_stop = EarlyStopping(patience=10)

# 模型初始化
transformer = Transformer(num_layers, d_model, num_heads, d_middle,
                          voc.num_words, voc.num_words,  # 多轮对话拆分为单轮让很多target同时也是input，且 train_set >> valid_set(强迫症同学自行分开input和output喔)
                          LCCC_train_set.max_length(), LCCC_train_set.max_length(),  # 所有对话中最长句子的长度(在Enc中根据每句话的长度对Position Matrix做了截断)，且我笃定train_set中的最长句子比valid_set中的长
                          )
if __name__ == '__main__':
    weight = torch.load(r'E:\AngryEcho\Try_Transformer\1_知乎从0开始系列\checkpoints\03.pth')
    transformer.load_state_dict(weight)
    for name, para in transformer.named_parameters():
        print(name, '\t', para)

transformer = transformer.cuda()
optimizer = optimize.Adam(transformer.parameters(), lr=LR)

# 避免由于可变序列长度或可变批次大小（最后一批次较小）导致的多次冗余转换
pass

# 记得model.train model.eval
MAX_EPOCHS = 1000
Train_Loss_List = []
Valid_Loss_List = []
for epoch in range(MAX_EPOCHS):

    # 获取loader: 注意这是个生成器！只允许被遍历一遍，第二次从头遍历的话，要重新生成一个新的迭代器
    train_loader = DataLoader(LCCC_train_set, voc, batch_size=BATCH_SIZE)
    valid_loader = DataLoader(LCCC_valid_set, voc, batch_size=BATCH_SIZE)

    Train_Loss = 0
    transformer.train()
    for input, target in train_loader:
        input, target = input.cuda(), target.cuda()
        loss = Train(input, target, transformer, Loss, loss_obj, optimizer)  #
        Train_Loss += loss
    print(f'Train Epoch[{epoch + 1}/{MAX_EPOCHS}]')
    Train_Loss_List.append(Train_Loss)

    Valid_Loss = 0
    with torch.no_grad():
        transformer.eval()
        for input, target in valid_loader:
            input, target = input.cuda(), target.cuda()
            loss = Valid(input, target, transformer, Loss, loss_obj)  # 循环的‘变量’是model 只有它在不断进化
            Valid_Loss += loss
    print(f'Epoch:{epoch + 1}  Train_Loss(Sum of one Epoch):{Train_Loss}  Valid_Loss(Sum of one Epoch):{Valid_Loss}')
    Valid_Loss_List.append(Valid_Loss)

    # if Valid_Loss < minimum_loss:
    #     minimum_loss = Valid_Loss
    #     torch.save(model.state_dict(), f'./Experiment_10/Model/50_1.pth')
    #     print('Saving Greater Model...')

    gc.collect()
    CUDA.empty_cache()

    early_stop(Valid_Loss)
    if early_stop.stop:
        torch.save(transformer.state_dict(), './checkpoints/03.pth')
        print(f'Model has been Saved at Epoch {epoch + 1}, and the optimal model has been saved')
        break

plot = Plot_loss(train_loss=Train_Loss_List, valid_loss=Valid_Loss_List)
plot.matplot_loss()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.savefig('./loss_curve.jpg')
plt.show()

print('Done!')
