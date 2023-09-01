class Vocab(object):
    """
    虽然有 bert-Chinese 字典，但小鱼不想用
    1. LCCC语料已经以’词‘为最小元分好了，如果用 ’字‘典 岂不是浪费？
    2. 造轮子计划，要造就造彻底
    所以在这里构造'词'典
    """
    def __init__(self, name, pad_token, sos_token, eos_token, unk_token):
        self.name = name
        self.pad_token = pad_token
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.trimmed = False
        self.word2index = {"PAD": pad_token, "SOS": sos_token, "EOS": eos_token, "UNK": unk_token}
        self.word2count = {}
        self.index2word = {pad_token: "PAD", sos_token: "SOS", eos_token: "EOS", unk_token: "UNK"}
        self.num_words = 4  # PAD SOS EOS UNK 是从0~3的四个词 下一个词的id是4

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        # 注意：
        # 训练语料中包含 PAD SOS EOS UNK 这四个'单'词（如果有iPAD没关系，因为它被作为一个词），然而它们一开始就作为“特殊词”被添加在self.word2index字典中
        # 所以当在语料中遇到这些词时不会把它们当作新词添加（进 if），而是直接跳转到 else
        # 由于没有经过if，self.word2count中没有这次词的Key，自然找不到，又如何添加呢？
        # 考虑到预料中出现特殊词会打破规则（比如某个人无意间说出EOS岂不是终止对话了？），我提前将语料中的特殊词替换成了小写
        else:
            self.word2count[word] += 1

    def addSentence(self, sentence):
        for word in sentence.split():  # 按空格切分词，适用于已分词的LCCC-base
            self.addWord(word)

    # 读入语料，建立词频
    def load_data(self, corpus_set):
        Single_Dialogs = corpus_set.Single_Dialogs  # 拆分为单轮后的语料的结构更简单，且语句没有缺失，所以词类、词频都是一样的，不影响做词典，何乐而不为呢？

        for dialog in Single_Dialogs:  # 一个列表是一轮对话
            for sentence in dialog:  # 一轮对话是两个句子（str）
                self.addSentence(sentence)

    # 将词典中词频过低的单词替换为unk_token
    # 需要一个代表修剪阈值的参数min_count，词频低于这个参数的单词会被替换为unk_token，相应的词典变量也会做出相应的改变
    def trim(self, min_count):
        if self.trimmed:  # 如果已经裁剪过了，那就直接返回
            return
        self.trimmed = True

        keep_words = []  # 保留的单词
        keep_num = 0  # 保留单词的数量
        for word, count in self.word2count.items():
            if count >= min_count:  # 对于所有应该保留的单词
                keep_num += 1
                # 后面是调用__addWord__来向词表中添加keep_word中的单词，而__addWord__中有略过重复词但词频数+1的机制，所以这里应该保留原本的样子
                for _ in range(count):
                    keep_words.append(word)
            else:
                pass

        print("剪枝后的词典密度: {} / {} = {:.4f}".format(keep_num, self.num_words - 4, keep_num / (self.num_words - 4)))

        # 重构词表
        self.word2index = {"PAD": self.pad_token, "SOS": self.sos_token, "EOS": self.eos_token, "UNK": self.unk_token}
        self.word2count = {}
        self.index2word = {self.pad_token: "PAD", self.sos_token: "SOS", self.eos_token: "EOS", self.unk_token: "UNK"}
        self.num_words = 4

        for word in keep_words:
            self.addWord(word)
