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

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self,x):
        x=x.view(-1,784) # -1其实就是自动获取mini_batch
        x=F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)  #最后一层不做激活，不进行非线性变换,交叉熵会自动做softmax

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
    print("epoch=",epoch)
    print("train loss=",total_loss/len(train_loader))

def test():
    correct=0
    total=0
    with torch.no_grad():
        for index,data in enumerate(test_loader,0):
            inputs,labels=data
            pre_y=model(inputs)
            _,pre_y=torch.max(pre_y.data,dim=1)#按行去最大的index
            total+=labels.size(0)
            correct+=(pre_y==labels).sum().item()
    print('accuracy on test =',100*correct/total)

if __name__=='__main__':
    for epoch in range(10):
        train(epoch)
        test()