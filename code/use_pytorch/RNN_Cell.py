import  torch
import torch.nn as nn

batch_size=1
input_size=4
hidden_size=4



"""
创建字典
设置输入数据中，每一个x的索引位置构成一个列表
同理。构建一个列表y
构建独热向量（one-hot）
利用for循环，搭建出每一个x对应的向量
-----
将搭建好的数据通过.view方法
构建张量
inputs（seq_len，batch,输入数据）（5，1，4）
label（seq_len,batch）
"""
idx2char=['e','h','l','o']
x_data=[1,0,2,2,3]
y_data=[3,1,2,3,2]
one_hot_lookup = [[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]]
x_one_hot=[one_hot_lookup[x] for x in x_data]
inputs=torch.Tensor(x_one_hot).view(-1,batch_size,input_size)
labels=torch.LongTensor(y_data).view(-1,1)##一定要加一维，确保loss函数的维度对齐

class Model(nn.Module):
    def __init__(self,input_size,hidden_size,batch_size):
        super(Model,self).__init__()
        self.rnn=nn.RNNCell(input_size,hidden_size)
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.batch_size=batch_size

    def forward(self,input,hidden):
        hidden=self.rnn(input,hidden)
        return hidden

    def init_hidden(self):
        return torch.zeros(self.batch_size,self.hidden_size)

model=Model(input_size,hidden_size,batch_size)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.1)

for epoch in range(10):
    """
        遍历10次
        损失值为更新为0
        梯度值为0
        取出训练集的特征和对应的标签 - 一次取一组，for循环来取
        将特征集填入模型，传出hidden-即输出值
        通过criterion函数计算损失值进行累加
        通过max函数取出，hidden中最大值，返回其索引位置 -- idx
        输出当前预测的结果

        反向传播
        优化器更新参数（权值，偏置，梯度）
        每次循环完，输出当前的循环的损失值
        """
    loss=0
    optimizer.zero_grad()
    hidden=model.init_hidden()
    for input,label in zip(inputs,labels):
        hidden=model(input,hidden)
        loss+=criterion(hidden,label) ##这里不用item是因为全部序列跑完就需要把loss全加算作一个epoch总loss
        _,idx=hidden.max(dim=1)
        print(idx2char[idx.item()],end="")
    loss.backward()
    optimizer.step()

    print(', Epoch [%d/15] loss = %.4f' % (epoch + 1, loss.item()))
