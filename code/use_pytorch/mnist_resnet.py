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

class res_block(torch.nn.Module):
    def __init__(self,in_chennal):
        super(res_block,self).__init__()
        self.channel=in_chennal
        self.conv1=torch.nn.Conv2d(in_chennal,in_chennal,kernel_size=3,padding=1)##残差要和原来的x相加所以通道数不能变
        self.conv2=torch.nn.Conv2d(in_chennal,in_chennal,kernel_size=3,padding=1)

    def forward(self,x):
        y=F.relu(self.conv1(x))
        y=x+self.conv2(y)
        return F.relu(y)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1=torch.nn.Conv2d(1,16,kernel_size=5)
        self.conv2=torch.nn.Conv2d(16,32,kernel_size=5)

        self.mp=torch.nn.MaxPool2d(2)

        self.linear=torch.nn.Linear(512,10)

        self.res1=res_block(16)
        self.res2=res_block(32)

    def forward(self,x):
        batch_size=x.size(0)
        x=F.relu(self.conv1(x))
        x=self.mp(x)
        x=self.res1(x)
        x=self.mp(F.relu(self.conv2(x)))
        x=self.res2(x)
        x=x.view(batch_size,-1)
        x=self.linear(x)
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