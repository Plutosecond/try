import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self,dropout=0.1):
        super(SelfAttention,self).__init__()
        self.dropout=nn.Dropout(dropout)
        self.softmax=nn.Softmax(-1)##对最后一个维度做softmax

    def forward(self,Q,K,V,mask=None):
        ##(batch,n_heads,seq_len,d_k)
        d_k=Q.size(-1)
        ##(batch,n_heads,seq_lenq,d_k)*(batch,n_heads,d_k,seq_lenk)->(batch,n_heads,seq_lenq,seq_lenk)
        scores=torch.matmul(Q,K.transpose(-2,-1))/math.sqrt(d_k)

        ##如果提供mask，则通过mask==0来找到需要屏蔽的位置，将他们改成-inf
        if mask is not None:
            scores=scores.masked_fill(mask==0,float('-inf'))

        attn=self.softmax(scores)
        attn=self.dropout(attn)##注意力权重
        ##(batch, n_heads, seq_lenq, seq_lenk)*(batch,n_heads,seq_lenv,d_k)->(batch,n_heads,seq_lenq,d_k)
        ##seq_k=seq_v
        out=torch.matmul(attn,V)##信息矩阵

        return out,attn

class MultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads,dropout=0.1):
        super(MultiHeadAttention,self).__init__()

        self.W_q=nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.dropout=nn.Dropout(dropout)
        self.num_heads=num_heads

        self.fc=nn.Linear(d_model,d_model)

        self.selfattn=SelfAttention(dropout)

        self.layernorm=nn.LayerNorm(d_model)##只输入一个参数默认最后一维，所以输入最后一维的维度

    def forward(self,q,k,v,mask=None):
        ##(batch_size,seq_len,d_model)
        batch_size=q.size(0)
        d_k=q.size(-1)
        d_k=d_k//self.num_heads
        ##(batch_size,seq_len,d_model)->(batch_size,seq_len,num_heads,d_k)->(batch_size,num_heads,seq_len,d_k)
        Q=self.W_q(q).view(batch_size,-1,self.num_heads,d_k).transpose(1,2)
        K=self.W_k(k).view(batch_size,-1,self.num_heads,d_k).transpose(1,2)
        V=self.W_v(v).view(batch_size,-1,self.num_heads,d_k).transpose(1,2)

        out,attn=self.selfattn(Q,K,V,mask)
        out=out.transpose(1,2).contiguous().view(batch_size,-1,self.num_heads*d_k)##(batch_size,seq_len,d_model)

        out=self.fc(out)
        out=self.dropout(out)

        return self.layernorm(out+q),attn

class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout=0.1):
        super(FeedForward,self).__init__()
        self.fc1=nn.Linear(d_model,d_ff)
        self.fc2=nn.Linear(d_ff,d_model)
        self.dropout=nn.Dropout(dropout)
        self.norm=nn.LayerNorm(d_model)

    def forward(self,x):
        out=self.dropout(F.relu(self.fc1(x)))
        out=self.fc2(out)
        return self.norm(x+out)

class EncoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout=0.1):
        super(EncoderLayer,self).__init__()

        self.multiheadattn=MultiHeadAttention(d_model,num_heads)
        self.feedforward=FeedForward(d_model,d_ff,dropout)

    def forward(self,x,mask=None):
        out,_=self.multiheadattn(x,x,x,mask)
        out=self.feedforward(out)
        return out ##out的形状和x形状一样（batch_size,seq_len,d_model)

class DecoderLayer(nn.Module):
    def __init__(self,d_model,num_heads,d_ff,dropout=0.1):
        super(DecoderLayer,self).__init__()
        self.self_attn=MultiHeadAttention(d_model,num_heads,dropout)
        self.cross_attn=MultiHeadAttention(d_model,num_heads,dropout)
        self.fc=FeedForward(d_model,d_ff,dropout)

    def forward(self,target,memory,target_mask=None,memory_mask=None):
        #target:目标序列，target_mask屏蔽未来的token，memory_mask：对padding的地方做掩码
        ##(batch_size,seq_len,d_model)
        out,self_attn=self.self_attn(target,target,target,target_mask)
        out,cross_attn=self.cross_attn(out,memory,memory,memory_mask)
        out=self.fc(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super(PositionalEncoding,self).__init__()
        # max_len=最大句子长度
        pe=torch.zeros(max_len,d_model)##初始化位置编码
        ##记录每个token位置的索引:0-maxlen-1
        ##[max_len,1]方便后续与缩放因子相乘
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        ##div_term每个维度得到缩放因子，torch.arange(0,d_model,2)生成偶数维度索引0，2，4对应2i
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        ##位置索引position*每个维度的缩放因子再套上sin得到位置编码
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        ##增加batch维度：1，maxlen，d_model，方便后续与输入embedding相加
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        ##(batch,seq_len,d_model)
        seq_len=x.size(1)
        ##取前seqlen的位置编码
        return x+self.pe[:,:seq_len,:]

class Encoder(nn.Module):
    def __init__(self,vocab_size,d_model,n_heads,num_layers,d_ff,dropout=0.1,max_len=5000):
        super(Encoder,self).__init__()
        self.embedding=nn.Embedding(vocab_size,d_model)
        self.pos_encoding=PositionalEncoding(d_model,max_len)
        ##堆叠多个encoderlayer
        self.encoder=nn.ModuleList([
            EncoderLayer(d_model,n_heads,d_ff,dropout) for _ in range(num_layers)
        ])

    def forward(self,inputs,mask=None):
        #乘上缩放因子，让后续注意力计算更稳定
        out=self.embedding(inputs)*math.sqrt(self.embedding.embedding_dim)
        out=self.pos_encoding(out)

        for layer in self.encoder:
            out=layer(out,mask)

        return out ##batch,seq_len,d_model

class Decoder(nn.Module):
    def __init__(self,vocab_size,d_model,n_heads,num_layers,d_ff,dropout=0.1,max_len=5000):
        super(Decoder,self).__init__()
        self.embedding=nn.Embedding(vocab_size,d_model)
        self.pos_encoding=PositionalEncoding(d_model,max_len)

        self.decoder=nn.ModuleList([
            DecoderLayer(d_model,n_heads,d_ff,dropout) for _ in range(num_layers)
        ])

        self.fc=nn.Linear(d_model,vocab_size)##要映射回原来的维度

    def forward(self,target,memory,target_mask=None,memory_mask=None):
        out=self.embedding(target)*math.sqrt(self.embedding.embedding_dim)
        out=self.pos_encoding(out)

        for layer in self.decoder:
            out=layer(out,memory,target_mask,memory_mask)

        return self.fc(out)

class Transformer(nn.Module):
    def __init__(self,
                 src_vocab,##原词表（特征维度）大小
                 tgt_vocab,##目标词表大小
                 d_model=512,
                 n_heads=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 d_ff=2048,
                 dropout=0.1,
                 max_len=5000):
        super(Transformer,self).__init__()
        self.encoder=Encoder(src_vocab,d_model,n_heads,num_encoder_layers,d_ff,dropout,max_len)
        self.decoder=Decoder(tgt_vocab,d_model,n_heads,num_decoder_layers,d_ff,dropout,max_len)

    def forward(self,src,tgt,src_mask=None,tgt_mask=None,memory_mask=None):
        ##src_mask用来屏蔽padding的地方
        memory=self.encoder(src,src_mask)
        ##tgt_mask用来屏蔽未来token
        out=self.decoder(tgt,memory,tgt_mask,memory_mask)
        return  out ##(batch,seq_len,tgt_vocab)

def generate_mask(size):
    mask=torch.triu(torch.ones(size,size),diagonal=1).bool()##生成上三角（不包括对角线
    return mask==0##生成下三角（包括对角线）

# 准备数据
idx2char = ['e', 'h', 'l', 'o']
src_data = [[1, 0, 2, 2, 3]]   # 输入: hello
tgt_data = [[3, 1, 2, 3, 2]]   # 目标: ohlol

src = torch.LongTensor(src_data)          # (1, 5)
tgt = torch.LongTensor(tgt_data)          # (1, 5)

# 训练时 Decoder 输入是 tgt[:, :-1]，预测 tgt[:, 1:]
tgt_input = tgt[:, :-1]                   # (1, 4)
tgt_output = tgt[:, 1:]                   # (1, 4)

# 生成 mask
tgt_mask = generate_mask(tgt_input.size(1)).unsqueeze(0).unsqueeze(0)  # (1,1,4,4)

# 模型
net = Transformer(src_vocab=4, tgt_vocab=4, d_model=16, n_heads=2,
                  num_encoder_layers=2, num_decoder_layers=2, d_ff=32, dropout=0.1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

# 训练
for epoch in range(10):
    optimizer.zero_grad()
    outputs = net(src, tgt_input, tgt_mask=tgt_mask)   # (1, 4, 4)
    loss = criterion(outputs.view(-1, 4), tgt_output.view(-1))
    loss.backward()
    optimizer.step()

    pred = outputs.argmax(dim=-1).squeeze(0).tolist()
    print(f"Epoch {epoch+1:3d}, loss = {loss.item():.4f}, pred = {''.join([idx2char[i] for i in pred])}")