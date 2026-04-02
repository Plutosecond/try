import torch
import torch.nn as nn
import numpy as np
import math

class TokenEmbedding(nn.Module):
    def __init__(self,c_in,d_model):
        super(TokenEmbedding,self).__init__()
        padding=1 if torch.__version__>='1.5.0' else 2
        self.tokenConv=nn.Conv1d(in_channels=c_in,out_channels=d_model,kernel_size=3,
                                 padding=3,padding_mode='circular',bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self,x):
        x=self.tokenConv(x.permute(0,2,1)).permute(0,2,1)
        return x

class FixedEmbedding(nn.Module):
    def __init__(self,c_in,d_model):
        super(FixedEmbedding,self).__init__()
        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()

class data_embedding(nn.Module):
    def __init__(self,c_in,d_model,dropout=0.1):
        super(data_embedding,self).__init__()

        self.value_embedding=TokenEmbedding(c_in,d_model)
        self.position_embedding=FixedEmbedding(c_in,d_model)

        self.dropout=nn.Dropout(p=dropout)

    def forward(self,x,x_mark):
        x=self.value_embedding(x)+self.position_embedding(x_mark)
        return self.dropout(x)

class moving_avg(nn.Module):
    def __init__(self,kernel_size,stride):
        super(moving_avg,self).__init__()
        self.kernel_size=kernel_size
        self.avg=nn.AvgPool1d(kernel_size=kernel_size,stride=stride,padding=0)

    def forward(self,x):
        front=x[:,0:1,:].repeat(1,self.kernel_size-1-math.floor((self.kernel_size-1)//2),1)#第一维度和第三维度不重复
        end=x[:,-1:,:].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x=torch.cat([front,x,end],dim=1)
        x=self.avg(x.permute(0,2,1))
        return x.permute(0,2,1)

class decomp(nn.Module):
    def __init__(self,kernel_size):
        super(decomp,self).__init__()
        self.moving_avg=moving_avg(kernel_size,stride=1)

    def forward(self,x):
        moving_mean=self.moving_avg(x)
        res=x-moving_mean
        return res,moving_mean

class Auto_attn(nn.Module):
    def __init__(self,factor=1):
        super(Auto_attn,self).__init__()
        self.factor=factor

    def time_delay_training(self,values,corr):
        head=values.shape[1]
        channel=values.shape[2]
        length=values.shape[3]

        top_k=int(self.factor*math.log(length))
        mean_value=torch.mean(torch.mean(corr,dim=1),dim=1)
        index=torch.topk(torch.mean(mean_value,dim=0),top_k,dim=-1)[1]
        weights=torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)

        tmp_corr=torch.softmax(weights,dim=-1)

        tmp_values=values
        delays_agg=torch.zeros_like(values).float()
        for i in range(top_k):
            pattern=torch.roll(tmp_values,-int(index[i]),-1)
            delays_agg=delays_agg+pattern* \
                       (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg# size=[B, H, d, S]

    def time_delay_inference(self,values,corr):
        batch=values.shape[0]
        head=values.shape[1]
        channel=values.shape[2]
        length=values.shape[3]

        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0).repeat(batch, head, channel, 1).cuda()
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights = torch.topk(mean_value, top_k, dim=-1)[0]
        delay = torch.topk(mean_value, top_k, dim=-1)[1]
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg


class AutoFormer(nn.Module):
    def __init__(self,configs):
        super(AutoFormer,self).__init__()
        self.seq_len=configs.seq_len
        self.pred_len=configs.pred_len
        self.enc_in=configs.enc_in
        self.dec_in=configs.dec_in
        self.d_model=configs.d_model

        self.enc_embedding=data_embedding(self.enc_in,self.d_model)
        self.dec_embedding=data_embedding(self.dec_in,self.d_model)

        self.enc_auto_attn=Auto_attn()
        self.dec_auto_attn=Auto_attn()

        self.decomp=decomp()
