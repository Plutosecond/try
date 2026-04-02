import torch
import torch.nn as nn
import math
import numpy as np

class my_Layernorm(nn.Module):
    """
    Special designed layernorm for the seasonal part
    """
    def __init__(self, channels):
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias

class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

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
        ##一层卷积核作为嵌入层
        self.dataConv=nn.Conv1d(c_in,d_model,kernel_size=3,padding=1,padding_mode='circular',bias=False)
        for m in self.modules():
            if isinstance(m,nn.Conv1d):##如果有一层卷积，就初始化权重
                nn.init.kaiming_normal_(m.weight,mode='fan_in',nonlinearity='leaky_relu')

        Embed_temporal=FixedEmbedding
        # self.m_embed = Embed_temporal(4, d_model)
        self.h_embed=Embed_temporal(24,d_model)
        self.weekday_embed=Embed_temporal(7,d_model)
        self.day_embed=Embed_temporal(32,d_model)
        self.month_embed = Embed_temporal(13, d_model)

        self.dropout=nn.Dropout(p=dropout)

    def forward(self,x,x_mark):
        data_embed=self.dataConv(x.permute(0,2,1)).transpose(1,2)
        # x_mark=x_mark.long()
        # # m_x=self.m_embed(x_mark[:,:,4]) if hasattr(self, 'minute_embed') else 0.
        # h_x=self.h_embed(x_mark[:,:,3])
        # wd_x=self.weekday_embed(x_mark[:,:,2])
        # d_x=self.day_embed(x_mark[:,:,1])
        # month_x=self.month_embed(x_mark[:,:,0])

        return self.dropout(data_embed)


class febattention(nn.Module):
    def __init__(self,seq_len,in_channels,out_channels):
        super(febattention,self).__init__()
        modes=min(seq_len//2,64)##seq_len//2是因为傅里叶变换是对称的
        index=list(range(0,seq_len//2))
        np.random.shuffle(index)
        self.index=index[:modes]
        self.index.sort()
        print('modes={}, index={}'.format(modes, self.index))

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
        self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.index), dtype=torch.cfloat))

    def forward(self,q,k,v):
        ##q[batch_size,seq_len,num_heads,head_dim]
        B,L,H,E=q.shape
        x=q.permute(0,2,3,1)##x[batch_size,num_heads,head_dim,seq_len]
        x_ft=torch.fft.rfft(x,dim=-1)##在最后一维做快速傅里叶变换(B,H,E,L//2+1)
        out_ft=torch.zeros(B,H,E,L//2+1,dtype=torch.cfloat)

        for wi,i in enumerate(self.index):
            out_ft[:,:,:,wi]=torch.einsum("bhi,hio->bho",x_ft[:,:,:,i],self.weights1[:,:,:,wi])

        x=torch.fft.irfft(out_ft,n=x.size(-1))
        return (x,None)


class moe(nn.Module):
    def __init__(self,kernel_size):
        super(moe,self).__init__()
        self.kernel_size=kernel_size
        self.avg=nn.AvgPool1d(kernel_size=kernel_size,stride=1,padding=0)

    def forward(self,x):
        ##x[batch_size, sequence_length, channels]
        front=x[:,0:1,:].repeat(1, self.kernel_size - 1-math.floor((self.kernel_size - 1) // 2), 1)
        end=x[:,-1:,:].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        mean=torch.cat([front,x,end],dim=1)
        mean=self.avg(mean.permute(0,2,1))
        mean=mean.permute(0,2,1)
        res=x-mean
        return res,mean

class fea(nn.Module):
    def __init__(self,in_channels,out_channels,q_seq_len,kv_seq_len):
        super(fea,self).__init__()
        ##取样
        q_modes=min(q_seq_len//2,64)
        kv_modes=min(kv_seq_len//2,64)
        q_index=list(range(0,q_modes))
        kv_index=list(range(0,kv_modes))
        np.random.shuffle(q_index)
        np.random.shuffle(kv_index)
        self.q_index=q_index[:q_modes]
        self.kv_index=kv_index[:kv_modes]
        self.q_index.sort()
        self.kv_index.sort()

        self.in_channels=in_channels
        self.out_channels=out_channels
        self.scale=1/(in_channels*out_channels)
        self.weights=nn.Parameter(
            self.scale * torch.rand(8, in_channels // 8, out_channels // 8, len(self.q_index), dtype=torch.cfloat))

    def forward(self,q,k,v):
        B,L,H,E=q.shape
        xq=q.permute(0,2,3,1)
        xk=k.permute(0,2,3,1)
        xv=v.permute(0,2,3,1)
        ##傅里叶变换+采样，v和k是一样的所以只要变换一个
        xq_ft_=torch.zeros(B,H,E,len(self.q_index),dtype=torch.cfloat)
        xq_ft=torch.fft.rfft(xq,dim=-1)
        for i,j in enumerate(self.q_index):
            xq_ft_[:,:,:,i]=xq_ft[:,:,:,j]

        xk_ft_=torch.zeros(B,H,E,len(self.kv_index),dtype=torch.cfloat)
        xk_ft=torch.fft.rfft(xk,dim=-1)
        for i,j in enumerate(self.kv_index):
            xk_ft_[:,:,:,i]=xk_ft[:,:,:,j]

        xqk_ft=(torch.einsum("bhex,bhey->bhxy",xq_ft_,xk_ft_))
        # if self.activation == 'tanh':
        xqk_ft = xqk_ft.tanh()
        # elif self.activation == 'softmax':
        #     xqk_ft = torch.softmax(abs(xqk_ft), dim=-1)
        #     xqk_ft = torch.complex(xqk_ft, torch.zeros_like(xqk_ft))
        xqkv_ft=(torch.einsum("bhxy,bhey->bhex",xqk_ft,xk_ft_))
        xqkvw=torch.einsum("bhex,heox->bhox",xqkv_ft,self.weights)
        ##padding
        out_ft=torch.zeros(B,H,E,L//2+1,dtype=torch.cfloat)
        for i, j in enumerate(self.q_index):
            out_ft[:, :, :, j] = xqkvw[:, :, :, i]

        ##多次矩阵乘法会积累很大数值，所以将out_ft先缩放一下
        out=torch.fft.irfft(out_ft / self.in_channels / self.out_channels, n=xq.size(-1))
        return (out,None)

class Multihead_attention(nn.Module):
    def __init__(self,attention,d_model,n_heads):
        super(Multihead_attention,self).__init__()

        self.attention=attention
        self.q_projection=nn.Linear(d_model,d_model)
        self.k_projection=nn.Linear(d_model,d_model)
        self.v_projection=nn.Linear(d_model,d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.n_heads=n_heads

    def forward(self,q,k,v):
        B,L,_=q.shape
        H=self.n_heads

        q=self.q_projection(q).view(B,L,H,-1)
        k=self.k_projection(k).view(B,L,H,-1)
        v=self.v_projection(v).view(B,L,H,-1)

        out,attn=self.attention(q,k,v)

        out=out.view(B,L,-1)
        return self.out_projection(out),attn


class EncoderLayer(nn.Module):
    def __init__(self,d_models,attention):
        super(EncoderLayer,self).__init__()
        self.attention=attention
        self.d_hidden=4*d_models
        self.conv1=nn.Conv1d(d_models,self.d_hidden,kernel_size=1,bias=False)
        self.conv2=nn.Conv1d(self.d_hidden,d_models,kernel_size=1,bias=False)
        self.decomp1=moe(kernel_size=25)
        self.decomp2=moe(kernel_size=25)

        self.dropout=nn.Dropout(0.1)
        self.activation=nn.functional.relu

    def forward(self,x):
        new_x,attn=self.attention(x,x,x)
        x=x+self.dropout(new_x)
        x,_=self.decomp1(x)
        y=x
        y=self.conv1(y.permute(0,2,1))
        y=self.dropout(self.activation(y))
        y=self.dropout(self.conv2(y).permute(0,2,1))
        res,_=self.decomp2(x+y)

        return res,attn

class encoder(nn.Module):
    def __init__(self,encoder_layers,norm_layer=None):
        super(encoder,self).__init__()
        self.encoder_layers=nn.ModuleList(encoder_layers)
        self.norm_layer=norm_layer

    def forward(self,x):
        attns=[]
        for encoder_layer in self.encoder_layers:
            x,attn=encoder_layer(x)
            attns.append(attn)

        if self.norm_layer is not None:
            x=self.norm_layer(x)

        return x,attns

class DecoderLayer(nn.Module):
    def __init__(self,d_models,feb,fea,c_out):
        super(DecoderLayer,self).__init__()
        d_hiddens=4*d_models
        self.decomp1=moe(kernel_size=25)
        self.decomp2=moe(kernel_size=25)
        self.decomp3=moe(kernel_size=25)

        self.conv1=nn.Conv1d(in_channels=d_models,out_channels=d_hiddens,kernel_size=1,bias=False)
        self.conv2=nn.Conv1d(in_channels=d_hiddens,out_channels=d_models,kernel_size=1,bias=False)
        self.activation=nn.functional.relu
        self.dropout=nn.Dropout(0.1)

        self.fea=fea
        self.feb=feb

        self.projection=nn.Conv1d(in_channels=d_models,out_channels=c_out,kernel_size=3,stride=1,padding=1,
                                  padding_mode='circular',bias=False)

    def forward(self,x,en_kv):
        newx=self.dropout(self.feb(x,x,x)[0])
        x=newx+x
        x,trend1=self.decomp1(x)
        x=x+self.dropout(self.fea(x,en_kv,en_kv)[0])
        x,trend2=self.decomp2(x)
        y=self.conv1(x.permute(0,2,1))
        y=self.dropout(self.activation(y))
        y=self.dropout(self.conv2(y).permute(0,2,1))
        x=x+y
        x,trend3=self.decomp3(x)

        trend=trend1+trend2+trend3
        trend=self.projection(trend.permute(0,2,1)).permute(0,2,1)

        return x,trend

class decoder(nn.Module):
    def __init__(self,decoder_layers,norm_layer=None,conv_layer=None):
        super(decoder,self).__init__()
        self.decoder_layers=nn.ModuleList(decoder_layers)
        self.norm_layer=norm_layer
        self.conv_layer=conv_layer

    def forward(self,x,en_kv,trend=None):
        for decoder_layer in self.decoder_layers:
            x,new_trend=decoder_layer(x,en_kv)
            trend=trend+new_trend

        if self.norm_layer is not None:
            x = self.norm_layer(x)

        if self.conv_layer is not None:
            x = self.conv_layer(x)
        return x, trend


class FEDmodel(nn.Module):
    def __init__(self,configs):
        super(FEDmodel,self).__init__()
        self.seq_len=configs.seq_len
        self.pred_len=configs.pred_len

        self.enc_embedding=data_embedding(configs.enc_in,configs.d_model)##数据嵌入
        self.dec_embedding=data_embedding(configs.dec_in,configs.d_model)

        self.decomp=moe(kernel_size=25)

        self.en_feb=febattention(self.seq_len,configs.d_model,configs.d_model)##feb代替自注意力
        self.dec_feb=febattention(self.seq_len//2+self.pred_len,configs.d_model,configs.d_model)
        self.fea=fea(configs.d_model,configs.d_model,self.seq_len//2+self.pred_len,self.seq_len)##fea代替解码器的注意力机制

        self.encoder=encoder(
            [
                EncoderLayer(
                    configs.d_model,
                    Multihead_attention(self.en_feb,configs.d_model,configs.n_heads))
                for l in range(configs.e_layers)
            ],norm_layer=my_Layernorm(configs.d_model)
        )

        self.decoder=decoder(
            [
                DecoderLayer(
                    configs.d_model,
                   Multihead_attention(self.dec_feb,configs.d_model,configs.n_heads),
                   Multihead_attention(self.fea,configs.d_model,configs.n_heads),
                    configs.c_out)
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            conv_layer=nn.Linear(configs.d_model,configs.c_out,bias=False)
        )

##x_enc[batch_size, sequence_length, channels]
    def forward(self,x_enc,x_enc_mask,x_dec_mask):
        ##encoder
        enc_out=self.enc_embedding(x_enc,x_enc_mask)
        enc_out,attns=self.encoder(enc_out)

        ##decoder
        seasonal,trend=self.decomp(x_enc)
        mean=torch.mean(x_enc,dim=1).unsqueeze(1).repeat(1,self.pred_len,1)
        trend=torch.cat([trend[:,-self.pred_len:,:],mean],dim=1)
        seasonal=nn.functional.pad(seasonal[:,-self.pred_len:,:],(0,0,0,self.pred_len))
        dec_out=self.dec_embedding(seasonal,x_dec_mask)
        seasonal_part,trend_part=self.decoder(dec_out,enc_out,trend)

        dec_out=trend_part+seasonal_part


        return dec_out[:,-self.pred_len:,:]

if __name__ == '__main__':
    class Configs(object):
        modes = 32
        moving_avg = [12, 24]
        seq_len = 96
        label_len = 48
        pred_len = 96
        enc_in = 7
        dec_in = 7
        d_model = 16
        freq = 'h'
        n_heads = 8
        d_ff = 16
        e_layers = 2
        d_layers = 1
        c_out = 7

    configs = Configs()
    model =FEDmodel(configs)

    print('parameter number is {}'.format(sum(p.numel() for p in model.parameters())))
    enc = torch.randn([3, configs.seq_len, 7])
    enc_mark = torch.randn([3, configs.seq_len, 4])

    # dec = torch.randn([3, configs.seq_len//2+configs.pred_len, 7])
    dec_mark = torch.randn([3, configs.seq_len//2+configs.pred_len, 4])
    out = model.forward(enc, enc_mark,dec_mark)
    print(out)