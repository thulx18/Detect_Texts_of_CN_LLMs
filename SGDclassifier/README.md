主要流程：

- 人、某两个商用大模型真实对话记录，各选8000数据，8:2切分训练集测试集

- jieba分词

- TfidfVectorizer统计ngrams特征，min-gram设置为1，max-gram设置为5

- 三个SGDClassifier集成学习

- 调了一些参数，整体准确率达到81.29%