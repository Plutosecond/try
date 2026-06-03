from torchvision import transforms
from torchvision import datasets
from torch.utils.data import Dataset,DataLoader
import torch
import torch.nn.functional as F
import numpy as np

batch_size=64

transforms=transforms.Compose([
    transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))
])

train_dataset = datasets.MNIST(root='dataset/mnist/', train=True, download=False, transform=transforms)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='dataset/mnist/', train=False, download=False, transform=transforms)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# (batch,1,28,28)

# torch.nn.Conv2d(1,10,kernel_size=3,stride=2,bias=False)
# 1是指输入的Channel，灰色图像是1维的；
# 10是指输出的Channel，也可以说第一个卷积层需要10个卷积核；
# kernel_size=3,卷积核大小是3x3；stride=2进行卷积运算时的步长，默认为1；
# bias=False卷积运算是否需要偏置bias，默认为False。
# padding = 0，卷积操作是否补0。
# self.fc = torch.nn.Linear(320, 10)，这个320获取的方式，可以通过x = x.view(batch_size, -1) # print(x.shape)可得到(64,320),64指的是batch，320就是指要进行全连接操作时，输入的特征维度。
class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)##一定要满足输入通道数等于原始通道数
        self.conv2=torch.nn.Conv2d(10,20,kernel_size=5)
        self.pooling=torch.nn.MaxPool2d(2)
        self.l1=torch.nn.Linear(320,160)
        self.l2=torch.nn.Linear(160,80)
        self.l3=torch.nn.Linear(80,10)

    def forward(self,x):
        batch_size=x.size(0)
        x=F.relu(self.pooling(self.conv1(x)))
        x=F.relu(self.pooling(self.conv2(x)))
        x=x.view(batch_size,-1)
        x=F.relu(self.l1(x))
        x=F.relu(self.l2(x))
        return self.l3(x)

model=Net()
criterion=torch.nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

def train(epoch):
    total_loss=0.0
    for index,data in enumerate(train_loader,0):
        inputs,labels=data
        pre_y=model(inputs)
        optimizer.zero_grad()
        loss=criterion(pre_y,labels)
        loss.backward()
        optimizer.step()

        total_loss+=loss.item()
    print(f'epoch={epoch},loss={total_loss/len(train_loader)}')

def test():
    correct=0
    total=0
    with torch.no_grad():
        for data in test_loader:
            inputs,labels=data
            pre_y=model(inputs)
            _,pre_y=torch.max(pre_y.data,dim=1)
            correct+=(pre_y==labels).sum().item()
            total+=labels.size(0)

    print(f'accuracy={100*correct/total}')

if __name__=='__main__':
    for epoch in range(10):
        train(epoch)
        test()