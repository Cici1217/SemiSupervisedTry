# _*_ coding:utf-8 _*_
# 利用深度学习做情感分析，基于Imdb 的50000个电影评论数据进行；
import datetime
import csv
import torch
from torch.utils.data import DataLoader, Dataset
import os
import re
from random import sample
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertTokenizer, DistilBertModel

# 个人理解
# 特征提取：BERT模型首先将字符串（文本数据）通过其内置的分词器转换为一系列整数（token
# IDs）。这个过程包括将每个单词或词片段映射到一个预定义词汇表中的索引。然后，BERT模型使用这些token
# IDs作为输入，提取文本数据的高级特征。
#
# BERT模型的作用：当数据通过BERT模型传递时，模型的多个层对这些特征进行进一步的处理，最终生成一个复杂的特征表示。这些特征捕捉了文本中的上下文信息，语义内容和语言结构等。
#
# 训练全连接层：在您的模型中，BERT模型的参数被冻结，这意味着它们在训练过程中不会更新。BERT模型的输出被传递到一个新添加的全连接层（通常是一个简单的神经网络层）。这个全连接层是可训练的，意味着它的参数（权重和偏置）会根据您的特定任务（在您的案例中是分类任务）进行调整。
#
# 输出分类结果：全连接层的输出是基于BERT提取的特征对文本进行分类的最终决策。例如，在二元分类任务中，这个层将输出两个值，分别对应于每个类别的预测分数。

# 存储训练出来的模型，现在是空的
model_path = "model"

# 读取数据的类
class ATISDataset(Dataset):
    def __init__(self, mode, trainNumber=4800, testNumber=800):
        super(ATISDataset, self).__init__()
        map0 = {
            'atis_flight': 0,
            'atis_flight_time': 1,
            'atis_airfare': 2,
            'atis_aircraft': 3,
            'atis_ground_service': 4,
            'atis_airline': 5,
            'atis_abbreviation': 6,
            'atis_quantity': 7
        }

        file_path = "archive/"
        if mode == "train":
            file_path += 'atis_intents_train.csv'
        if mode == "test":
            file_path += 'atis_intents_test.csv'
        string_array = []
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                if len(string_array) == 4800:
                    break
                string_array.append(row)

        self.total_file = []
        self.total_label = []
        for s in string_array:
            self.total_file.append(s[1])
            self.total_label.append(map0.get(s[0]))



    def tokenize(self, text):
        # 具体要过滤掉哪些字符要看你的文本质量如何
        # 这里定义了一个过滤器，主要是去掉一些没用的无意义字符，标点符号，html字符啥的
        fileters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                    '\?', '@'
            , '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', ]
        # sub方法是替换
        text = re.sub("<.*?>", " ", text, flags=re.S)  # 去掉<...>中间的内容，主要是文本内容中存在<br/>等内容
        text = re.sub("|".join(fileters), " ", text, flags=re.S)  # 替换掉特殊字符，'|'是把所有要匹配的特殊字符连在一起
        return text  # 返回文本

    def __getitem__(self, idx):
        labels = []
        sentences = []
        labels.append(self.total_label[idx])
        sentences.append(self.total_file[idx])
        return sentences, labels

    def __len__(self):
        return len(self.total_file)


# nn.Module 是所有神经网络模块的基类，本类继承这个类
# 使用DistilBERT模型和分词器。基于这个模型，进行训练
class BertClassificationModel(nn.Module):
    def __init__(self, hidden_size=768):
        # hidden_size：模型的隐藏层大小，对应于DistilBERT模型的输出特征维度
        super(BertClassificationModel, self).__init__()
        # 使用 DistilBERT
        model_name = 'phi-2'

        # 使用 DistilBERT 的分词器和模型
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.bert = DistilBertModel.from_pretrained(model_name)


        # 冻结bert参数，因为我要训练的是新的全连接层，不用训练预训练模型。这样做可以防止破坏预训练模型的特点
        # for p in self.bert.parameters():
        #     p.requires_grad = False
        self.fc = nn.Linear(hidden_size, 8)

    # 这个方法代表数据的流动：数据在神经网络模型中的传递过程，从输入层开始，经过一系列变换，最终到达输出层并生成结果
    def forward(self, batch_sentences):
        # 这里的sentences_tokenizer用来处理输入的文本数据。它将文本转换为模型可以理解的格式。
        # 分词（Tokenization）：将文本拆分为tokens，如单词
        # 添加特殊标记（Special Tokens）：如[CLS]（开始）和[SEP]（分个句子），这些对于BERT模型是必要的。
        # 转换为词汇表索引（Token to Index Mapping）：将每个token转换为词汇表中的索引。
        # 填充（Padding）和截断（Truncation）：确保所有输入序列长度一致，以满足模型的输入要求。
        sentences_tokenizer = self.tokenizer(batch_sentences,
                                             truncation=True,
                                             padding=True,
                                             max_length=512,
                                             add_special_tokens=True)

        # 分词器输出的词汇表索引
        input_ids = torch.tensor(sentences_tokenizer['input_ids'])
        # 区分真实的token和填充的token
        attention_mask = torch.tensor(sentences_tokenizer['attention_mask'])
        # 转换后的特征表示
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = bert_out[0]
        # [batch_size, sequence_length, hidden_size] ，BERT模型输出的最后一层隐藏状态
        bert_cls_hidden_state = last_hidden_state[:, 0, :]
        # 每个序列（就是一个评论）的第一个token（通常是 [CLS] token）的隐藏状态，代表这个评论的state
        fc_out = self.fc(bert_cls_hidden_state)
        # 该state被传递到一个全连接层（self.fc），该层负责将这些特征映射到最终的分类结果上
        # 总的来说，全连接层处理特征，生成结果

        return fc_out
    # 返回最终分类结果

def main():
    trainNumber = 15611
    testNumber = 800
    # 定义每次放多少个数据参加训练
    batchsize = 200

    trainDatas = ATISDataset(mode="train", trainNumber=trainNumber)
    testDatas = ATISDataset(mode="test", testNumber=testNumber)

    # 遍历train_loader/test_loader 每次返回batch_size条数据
    train_loader = torch.utils.data.DataLoader(trainDatas, batch_size=batchsize, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testDatas, batch_size=batchsize, shuffle=False)

    print('模型数据已经加载完成，预训练评估')

    now = datetime.datetime.now()
    print(now)

    # 初始化模型
    model = BertClassificationModel()

    # 初始化评估
    model.eval()


    num = 0
    # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化,主要是在测试场景下使用
    tempNum = 1
    for j, (data, labels) in enumerate(test_loader, 0):
        output = model(data[0])
        out = output.argmax(dim=1)
        num += (out == labels[0]).sum().item()
        print('Temp Accuracy:', num / ((tempNum) * batchsize))
        tempNum = tempNum + 1
    print('Accuracy:', num / testNumber)

    now = datetime.datetime.now()
    print(now)

    # 首先定义优化器，这里用的AdamW，lr是学习率，因为bert用的就是这个
    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    # optimizer = optim.AdamW([
    #     {'params': model.bert.parameters(), 'lr': 2e-5},  # 对BERT模型的参数使用较小的学习率
    #     {'params': model.fc.parameters(), 'lr': 1e-3}  # 对新加全连接层的参数使用较大的学习率
    # ])

    # 这里是定义损失函数，交叉熵损失函数，常用解决分类问题
    criterion = nn.CrossEntropyLoss()

    print("预测试已完成,现在开始模型训练。")
    now = datetime.datetime.now()
    print(now)

    # 这里搭建训练循环，输出训练结果
    # 设置循环多少次训练
    epoch_num = 5
    # 调整model为train模式
    model.train()
    # 循环训练

    # 在训练过程中，将训练文本数据批量传递给模型。对于每个批次enumerate(train_loader, 0)，PyTorch自动调用 forward 方法来计算输出（output）。
    # 损失计算与梯度反向传播：接着，根据模型的输出和真实标签计算损失（例如，使用交叉熵损失）。然后，使用梯度反向传播（loss.backward()）来计算网络参数的梯度。
    # 参数更新：最后，使用优化器（例如Adam）更新模型的权重，这通常涉及到更新全连接层的参数，BERT模型是冻结的。
    for epoch in range(epoch_num):
        for i, (data, labels) in enumerate(train_loader, 0):
            output = model(data[0])
            optimizer.zero_grad()  # 梯度清0
            loss = criterion(output, labels[0])  # 计算误差
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数

            # 打印一下每一次数据扔进去学习的进展
            print('batch:%d loss:%.5f' % (i, loss.item()))

        # 打印一下每个epoch的深度学习的进展i
        print('epoch:%d loss:%.5f' % (epoch, loss.item()))

    now = datetime.datetime.now()
    print(now)

    # 下面开始测试模型是不是好用哈
    print('新模型准确率验证')
    now = datetime.datetime.now()
    print(now)

    # 这里载入验证模型，他把数据放进去拿输出和输入比较，然后除以总数计算准确率
    # 鉴于这个模型非常简单，就只用了准确率这一个参数，没有考虑混淆矩阵这些
    num = 0
    tempNum = 1
    model.eval()  # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化,主要是在测试场景下使用；
    for j, (data, labels) in enumerate(test_loader, 0):
        output = model(data[0])
        out = output.argmax(dim=1)
        print(out == labels[0])
        num += (out == labels[0]).sum().item()
        print('Temp Accuracy:', num / ((tempNum) * batchsize))
        tempNum = tempNum + 1
    print('Accuracy:', num / testNumber)

    now = datetime.datetime.now()
    print(now)

    print('保存模型')
    # 保存模型
    torch.save(model.state_dict(), os.path.join(model_path, 'bert_sentiment_analysis.pth'))


if __name__ == '__main__':
    main()
