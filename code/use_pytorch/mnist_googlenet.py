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

class Inception_block(torch.nn.Module):
    def __init__(self,in_channels):
        super(Inception_block,self).__init__()
        self.branch_pool=torch.nn.Conv2d(in_channels,24,kernel_size=1)##1*1的卷积核可以改变通道数

        self.branch1x1=torch.nn.Conv2d(in_channels,16,kernel_size=1)

        self.branch5x5_1=torch.nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch5x5_2=torch.nn.Conv2d(16,24,kernel_size=5,padding=2)

        self.branch3x3_1=torch.nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2=torch.nn.Conv2d(16,24,kernel_size=3,padding=1)
        self.branch3x3_3=torch.nn.Conv2d(24,24,kernel_size=3,padding=1)

    def forward(self,x):
        branch1x1=self.branch1x1(x)

        branch5x5=self.branch5x5_1(x)
        branch5x5=self.branch5x5_2(branch5x5)

        branch3x3=self.branch3x3_1(x)
        branch3x3=self.branch3x3_2(branch3x3)
        branch3x3=self.branch3x3_3(branch3x3)

        branch_pool=F.avg_pool2d(x,kernel_size=3,stride=1,padding=1)
        branch_pool=self.branch_pool(branch_pool)

        outputs=[branch_pool,branch1x1,branch3x3,branch5x5]
        return torch.cat(outputs,dim=1) #b,c,w,h 按channel合并，dim=1,输出的channel数为24+24+24+16=88

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(88,20,kernel_size=5)

        self.incep1=Inception_block(10)
        self.incep2=Inception_block(20)

        self.mp=torch.nn.MaxPool2d(2)
        self.fc=torch.nn.Linear(1408,10)

    def forward(self,x):#b,c,w,h
        in_size=x.size(0)
        x=F.relu(self.mp(self.conv1(x)))
        x=self.incep1(x)
        x=F.relu(self.mp(self.conv2(x)))
        x=self.incep2(x)
        x=x.view(in_size,-1)##(in_size,1408)
        x=self.fc(x)
        return x

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
            total+=labels.size(0)
            correct+=(pre_y==labels).sum().item()
    print(f'accuracy on test={100*correct/total},total={total},correct={correct}')

if __name__=='__main__':
    for epoch in range(10):
        train(epoch)
        test()