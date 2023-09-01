from TextLoader import load_json
from torch.utils.data import Dataset

# 分词器 We don't need !
"""
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained(r'./bert-base-chinese')
seq = '鱼小冬是世界上最傻的人'
print(tokenizer(seq))
print(tokenizer(seq, add_special_tokens=False))
print(tokenizer(seq, add_special_tokens=False, max_length=7, truncation=True))
print(tokenizer(seq, add_special_tokens=False, max_length=15, padding='max_length'))
print()
tokens = tokenizer.tokenize(seq)
ids_1 = tokenizer.convert_tokens_to_ids(tokens)
ids_2 = tokenizer.encode(seq)
strs = tokenizer.convert_tokens_to_string(tokens)
back_1 = tokenizer.convert_ids_to_tokens(ids)
back_2 = tokenizer.decode(ids)
print(tokens)
print(ids_1)
print(ids_2)
print(strs)
print(back_1)
print(back_2)
"""

# json
'''
test_set = load_json('./dataset/LCCC-base-split/LCCC-base_test.json')
print(type(test_set))
print(test_set[0])
print(test_set[0][0])
print(test_set[0][0][0])
'''


class corpus_set(Dataset):
    def __init__(self, path):
        Dialogs = load_json(path)

        self.Single_Dialogs = []
        # 将多轮对话拆分为连续的单轮对话
        # 本来还想将多轮对话的前几句合并为一句，但是研究了一下语料觉得这样没道理
        # 因为每次都是你一句我一句，没有连续两句来自同一个人，所以合并为一句不通顺
        # 拆分后虽然是单轮对话，但同一个时间段内（同一个语境下）的对话是连续的。提问：这会对模型理解对话或学习有帮助吗？(ˉ▽ˉ；)...
        for dialog in Dialogs:
            max_len = len(dialog)

            if max_len > 2:
                for i in range(max_len - 1):  # 最少两句才能形成一轮对话，所以最后一句只能作为最后一轮对话的答复语句
                    self.Single_Dialogs.append(dialog[i:i + 2])  # 不包含右界
            else:
                self.Single_Dialogs.append(dialog)

        del Dialogs

        # 拆分后的语料从多轮25万变为单轮80万，太大了。考虑到拆分多轮后的单轮是连续的（有上下文联系），这里有两种稀疏方法
        # self.Single_Dialogs = self.Single_Dialogs[::2]  # 打破连续的单轮
        self.Single_Dialogs = self.Single_Dialogs[:len(self.Single_Dialogs) // 200]  # 取整体的前_分之一

    def __getitem__(self, index):
        input_seq = self.Single_Dialogs[index][0]
        target_seq = self.Single_Dialogs[index][1]
        return input_seq, target_seq

    def __len__(self):
        return len(self.Single_Dialogs)

    def max_length(self):
        sentence_length = []

        for dialog in self.Single_Dialogs:
            for sentence in dialog:
                sentence_length.append(len(sentence))

        return max(sentence_length)


if __name__ == '__main__':
    train_set = corpus_set(r'E:\AngryEcho\Try_Transformer\1_知乎从0开始系列\dataset\LCCC-base-split\LCCC-base_train.json')
    for i, j in train_set:
        print(i, '\t', j)
