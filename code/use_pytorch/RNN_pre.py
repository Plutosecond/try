import torch

batch_size=1
seq_len=3
input_size=4
hidden_size=2

numlayers=2

cell=torch.nn.RNNCell(input_size,hidden_size)

rnn=torch.nn.RNN(input_size,hidden_size,num_layers=numlayers)##使用RNN得多一个numlayers

#(seq, batch, features)
"""
生成标准正态分布的（均值为零，方差为一的高斯白噪声）的随机数；
.randn生成（3，1，4）  --  张量
生成为全为0.的数，
.zeros生成（1，2）  --  张量
"""

dataset=torch.randn(seq_len,batch_size,input_size)
hidden_init=torch.zeros(batch_size,hidden_size)

for idx,data in enumerate(dataset):
    """
        idx：第几次遍历
        data：取出当前次数的数据，数据维度是（1，4）
        cell（输入数据（batch，特征的向量维度），记忆体维度（batch，hidden的维度））
     """
    print('='*20,idx,'RNN_Cell','='*20)
    hidden_init=cell(data,hidden_init)
    print(hidden_init)



"""
循环神经网络函数RNN
（输入数据（数据的时间维度x的个数，batch，单个数据的维度），记忆体数据（层数（也即第几层的hidden），hidden的维度），层数）
"""
hidden=torch.zeros(numlayers,batch_size,hidden_size)##使用RNN得多一个numlayers

"""
输出数据：
out：（seq_len，batch，hidden数据维度）--最上边的那一行，横着的，也就是输出
hidden：（层数，batch，hidden数据维度）最右边一行
"""

out,hidden=rnn(dataset,hidden)
print('Output size: ', out.shape)
print('Output: ', out)
print('Hidden size: ', hidden.shape)
print('Hidden: ', hidden)