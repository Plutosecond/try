import torch
import torch.nn.functional as F

x_data=torch.Tensor([[1.0],[2.0],[3.0]])
y_data=torch.Tensor([[3.0],[5.0],[7.0]])

x_log=torch.Tensor([[1.0],[2.0],[3.0]])
y_log=torch.Tensor([[0],[0],[1]])

class LinearModel(torch.nn.Module):##Module自动实现反向传播
    def __init__(self):
        super(LinearModel,self).__init__()
        self.linear=torch.nn.Linear(1,1)##（输入特征数，输出特征数）他会自动匹配权重矩阵

    def forward(self,x):
        y_pred=self.linear(x)
        return y_pred

model=LinearModel()

criterion=torch.nn.MSELoss(size_average=False)
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)##把model里需要算梯度的参数都传进去

# for epoch in range(100):
#     y_pred=model(x_data)##1、计算预测值
#     loss=criterion(y_pred,y_data)##2、计算loss
#     # print(epoch,loss)
#     optimizer.zero_grad()##梯度清零（可选）
#     loss.backward()##3、backward
#     optimizer.step()##4、更新权重
#     print(epoch)
#     print("w=",model.linear.weight.item())
#     print("b=",model.linear.bias.item())
#
# x_test=torch.Tensor([4.0])
# y_test=model(x_test)
# print("y_pred=",y_test.data)

class Linear_with_logistic_reg(torch.nn.Module):
    def __init__(self):
        super(Linear_with_logistic_reg,self).__init__()
        self.linear=torch.nn.Linear(1,1)

    def forward(self,x):##加上sigmoid，逻辑回归
        y_pred=F.sigmoid(self.linear(x))
        return y_pred

log_model=Linear_with_logistic_reg()
log_criterion=torch.nn.BCELoss(size_average=False)
log_optimizer=torch.optim.SGD(log_model.parameters(),lr=0.01)

for epoch in range(100):
    y_pred=log_model(x_log)
    loss=log_criterion(y_pred,y_log)
    print(epoch, loss)
    log_optimizer.zero_grad()
    loss.backward()
    log_optimizer.step()

x_log_test=torch.Tensor([4.0])
y_test=log_model(x_log_test)
print("y_pred=",y_test.data)