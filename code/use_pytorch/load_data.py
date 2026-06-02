import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df=pd.read_csv('diabetes.csv')
x=df.drop('Outcome',axis=1).values##axis=1是删除列，=0是删除行
y=df['Outcome'].values

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)


class DiabetesDataset(Dataset):
    def __init__(self,features,labels):
        self.features=torch.tensor(features,dtype=torch.float32)
        self.labels=torch.tensor(labels,dtype=torch.float32).view(-1,1)
    def __getitem__(self, index):
        return self.features[index],self.labels[index]

    def __len__(self):
        return len(self.features)

train_dataset=DiabetesDataset(xtrain,ytrain)
test_dataset=DiabetesDataset(xtest,ytest)

train_loader=DataLoader(dataset=train_dataset,batch_size=32,shuffle=True,num_workers=0)
test_loader=DataLoader(dataset=test_dataset,batch_size=32,shuffle=False,num_workers=0)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model,self).__init__()
        self.linear1=torch.nn.Linear(8,6)
        self.linear2=torch.nn.Linear(6,4)
        self.linear3=torch.nn.Linear(4,2)
        self.linear4=torch.nn.Linear(2,1)
        self.sigmoid=torch.nn.Sigmoid()

    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        return x

model=Model()

criterion=torch.nn.BCELoss(reduction='mean')
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

def train(epoch):
    model.train()
    train_loss=0.0
    for i,data in enumerate(train_loader,0):
        inputs,labels=data
        y_pred=model(inputs)
        loss=criterion(y_pred,labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        train_loss+=loss.item()
    if epoch%20==0:
        print("train loss=",train_loss/len(train_loader))

def test(epoch):
    model.eval()
    correct=0
    total=0
    test_loss=0.0
    with torch.no_grad():
        for inputs,labels in test_loader:
            outputs=model(inputs)
            loss=criterion(outputs,labels)

            test_loss+=loss.item()
            predicted=(outputs>0.5).float()
            total+=labels.size(0)##一个batch的样本数
            correct+=(predicted==labels).sum().item()

    avg_loss=test_loss/len(test_loader)##len（test_loader)是batch数量
    accuracy=100*correct/total
    print(f"Epoch {epoch}, Test Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

if __name__=='__main__':
    for epoch in range(200):
        train(epoch)
        if epoch%20==0:
            test(epoch)