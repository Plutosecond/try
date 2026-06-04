import torch
import torch.nn as nn

num_class=4
batch_size=1
seq_len=5
input_size=4
hidden_size=8
num_layers=2
embedding_size=10


"""
创建字典，创建由特征索引构成的二维向量（1,5），和对应的标签对应字母索引所构成的（5）
将x_data y_data转换为张量  （1，5）（5）
"""
idx2char = ['e', 'h', 'l', 'o']
x_data = [[1, 0, 2, 2, 3]]
y_data = [3, 1, 2, 3, 2]
inputs = torch.LongTensor(x_data)
labels = torch.LongTensor(y_data)

class Model(nn.Module):
    def __init__(self,input_size,hidden_size,batch_size,num_layers,num_class):
        super(Model,self).__init__()
        self.emb=nn.Embedding(input_size,embedding_size)
        self.rnn=nn.RNN(embedding_size,hidden_size,num_layers,batch_first=False)

        self.fc=nn.Linear(hidden_size,num_class)

        self.batch_size=batch_size
        self.num_layers=num_layers
        self.hidden_size=hidden_size

    def init_hidden(self):
        return torch.zeros(self.num_layers,self.batch_size,self.hidden_size)

    def forward(self,inputs,hidden):
        """
                先对输入数据进行Embedding，嵌入层，将input_size  --  embedding_size；得到高维数据---(batch_size,seq_len,embedding_size)
                将得到的数据进行rnn模型进行训练（x（seq_len,batch_size,embedding_size），hidden（num_layers,batch,hidden_size））
                返回out最后一层的输出，和 hidden最后时刻的记忆体的参数
                out:(seq_len,batch,hiddensize)
                hide:(num_layer,batch_size, hidden_size)
                通过线性函数Linear。维度是num_class
                .view()展示的是将（out即x）的每一个值都组合起来，变成（seq_len，num_class）batch_size
                """
        inputs=self.emb(inputs)##(batch_size,seq_len,embedding_size)
        inputs=inputs.permute(1,0,2)##重排维度，或者把上面batchfirst设置为True也可以
        # print(inputs.size())
        outputs,hide=self.rnn(inputs,hidden)
        outputs=self.fc(outputs)##nn.Linear 会自动将输入的最后一维 进行线性变换，其他维度保持不变
        return outputs.view(-1,num_class)

net=Model(input_size,hidden_size,batch_size,num_layers,num_class)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(15):
    """
    优化器的梯度设置为0
    将input放入模型，返回out的数据（是矩阵（seq_len，num_class））
    计算损失函数
    反向传播
    优化器迭代，更新参数（权重，偏置，梯度）
    通过max返回每一组数据的最大值对应的索引
    查找出字典对应索引的字母，输出
    输出当前迭代次数和损失值
    """
    # loss = 0
    hidden=net.init_hidden()
    optimizer.zero_grad()
    outputs = net(inputs,hidden)
    print(outputs.shape)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()

    print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
    print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))