import torch.nn as nn
import torch
import math
import numpy as np

#点积注意力机制
class ScaleDotProductAttention(nn.Module):
  def __init__(self,d_k):
    #调用父类nn.module的初始化方法，两个参数一个当前类一个当前类的实例
    #执行父类的初始化再执行子类的初始化
    super(ScaleDotProductAttention,self).__init__()
    self.d_k=d_k

  def forward(self,q,k,v,attention_mask=None):
    # q: [batch_size, n_heads, len_q, d_k]
    # k: [batch_size, n_heads, len_k, d_k]
    # v: [batch_size, n_heads, len_v, d_v]
    # attn_mask: [batch_size, n_heads, seq_len, seq_len]
    '''matmul可用于批量矩阵乘法，transpose交换维度，实现转置'''
    #scores: [batch_size, n_heads, len_q, len_q]
    scores=torch.matmul(q,k.transpose(-1,-2))/math.sqrt(self.d_k)

    '''掩码的作用！输入序列通常会被填充到相同的长度以便批量处理，引入掩码使它们不参与计算；同时掩码可控制模型可见范围'''
    scores=scores.masked_fill(attention_mask==0,-1e9)#把被mask的地方置为无限小，softmax之后基本就是0，也就对q不起作用
    attn=nn.functional.softmax(scores,dim=-1)
    context=torch.matmul(attn,v)
    return context,attn

'''多头注意力层：每个头拼在一起，然后接一个线性层映射'''
#多个头揉在一起，n_heads
class MultiHeadAttention(nn.Module):
  def __init__(self,d_model,d_k,d_v,n_heads):
    super(MultiHeadAttention,self).__init__()
    self.d_model=d_model
    self.d_k=d_k
    self.d_v=d_v
    self.n_heads=n_heads
    '''可训练的qkv矩阵，详情见知乎说明'''
    self.w_q=nn.Linear(d_model,d_k*n_heads,bias=False)
    self.w_k=nn.Linear(d_model,d_k*n_heads,bias=False)
    self.w_v=nn.Linear(d_model,d_v*n_heads,bias=False)
    self.fc=nn.Linear(n_heads*d_v,d_model,bias=False)#映射回输入维度
    self.layernorm=nn.LayerNorm(d_model)#层归一化在神经网络中用于稳定和加速训练过程

  def forward(self,q,k,v,attention_mask=None):
    # q: [batch_size, seq_len, d_model]
    # k: [batch_size, seq_len, d_model]
    # v: [batch_size, seq_len, d_model]
    # attn_mask: [batch_size, seq_len, seq_len]

    residual,batch_size=q,q.size(0) #residual存原始值
    '''view调整张量形状'''
    #q: [batch_size, n_heads, len_q, d_k]
    q=self.w_q(q).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
    k=self.w_k(k).view(batch_size,-1,self.n_heads,self.d_k).transpose(1,2)
    v=self.w_v(v).view(batch_size,-1,self.n_heads,self.d_v).transpose(1,2)
    #attn_mask : [batch_size, n_heads, seq_len, seq_len]
    '''unsqueeze加一维，repeat复制多份，使得维度匹配和前面一样'''
    attention_mask=attention_mask.unsqueeze(1).repeat(1,self.n_heads,1,1)
    #context: [batch_size, n_heads, len_q, d_v]
    context,attn=ScaleDotProductAttention(self.d_k)(q,k,v,attention_mask)
    #context: [batch_size, len_q, n_heads * d_v]
    '''前面n个头同时做，这里出来把他们拼接'''
    context=context.transpose(1,2).reshape(batch_size,-1,self.n_heads*self.d_v)
    #还原回去
    output=self.fc(context)
    '''transformer示意图中的——ADD&Norm'''
    return self.layernorm(output+residual),attn
  
#前馈神经网络部分，transformer示意图中的Feed Forward 两个线性全连接层
class PositionwiseFeedForwardNet(nn.Module):
  def __init__(self,d_model,d_ff):  
    super(PositionwiseFeedForwardNet,self).__init__()
    self.fc=nn.Sequential(
      nn.Linear(d_model,d_ff),
      nn.ReLU(),
      nn.Linear(d_ff,d_model)
    )
    '''再次强调层归一化的作用'''
    self.layernorm=nn.LayerNorm(d_model)

  def forward(self,x):
    #x：[batch_size, seq_len, d_model]
    residual=x
    output=self.fc(x)
    #[batch_size, seq_len, d_model]
    return self.layernorm(output+residual)
  
#解码层
class DecoderLayer(nn.Module):
  def __init__(self,d_model,d_k,d_v,d_ff,n_heads):
    super(DecoderLayer,self).__init__()
    '''和前面一样的多头和前馈'''
    self.attention=MultiHeadAttention(d_model,d_k,d_v,n_heads)
    self.pos_ffn=PositionwiseFeedForwardNet(d_model,d_ff)

  def forward(self,inputs,attention_mask):
    # inputs: [batch_size, seq_len, d_model]
    #outputs: [batch_size, seq_len, d_model]
    #attention_mask: [batch_size, seq_len, seq_len]
    outputs,self_attn=self.attention(inputs,inputs,inputs,attention_mask)#qkv相同，都是inputs，编码层的输出
    outputs=self.pos_ffn(outputs)
    return outputs,self_attn
  
#解码器
class PositionalEncoding(nn.Module):
  def __init__(self,d_model,max_pos,device):
    super(PositionalEncoding,self).__init__()
    self.device=device
    '''embedding的本质：将离散的索引映射为连续的向量，生成位置编码'''
    '''比如：
        位置索引	位置编码（向量）
        0	[0.1, 0.2, 0.3, 0.4]
        1	[0.5, 0.6, 0.7, 0.8]
        2	[0.9, 1.0, 1.1, 1.2]
        3	[1.3, 1.4, 1.5, 1.6]
        4	[1.7, 1.8, 1.9, 2.0]
        '''
    self.pos_embedding=nn.Embedding(max_pos,d_model)

  def forward(self,inputs):
    seq_len=inputs.size(1)
    #arange:等差数列的一维张量，用于前面位置索引
    pos=torch.arange(seq_len,dtype=torch.long,device=self.device)
    #[seq_len] -> [batch_size, seq_len]
    pos=pos.unsqueeze(0).expand_as(inputs)
    return self.pos_embedding(pos)


'''对于Transformer Decoder结构，模型在解码时应该是自回归的，每次都是基于之前的信息预测下一个Token，这意味着在生成序列的第 i 个元素时，模型只能看到位置 i 之前的信息。因此在训练时需要进行遮盖，防止模型看到未来的信息，遮盖的操作也非常简单，可以构建一个上三角掩码器。'''
class Decoder(nn.Module):
  def __init__(self, d_model, n_heads, d_ff, d_k, d_v, vocab_size, max_pos, n_layers, device):
        super(Decoder, self).__init__()
        self.device = device
        # 将Token转为向量
        self.embedding = nn.Embedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_pos, device)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, d_k, d_v) for _ in range(n_layers)])

  def forward(self, inputs, attention_mask):
        ##
        # inputs: [batch_size, seq_len]
        ##
        # [batch_size, seq_len, d_model]
        outputs = self.embedding(inputs) + self.pos_encoding(inputs)
        # 上三角掩码，防止看到未来的信息， [batch_size, seq_len, seq_len]
        subsequence_mask = self.get_attn_subsequence_mask(inputs, self.device)
        if attention_mask is not None:
            # pad掩码 [batch_size, seq_len, seq_len]
            attention_mask = self.get_attn_pad_mask(attention_mask)
            # [batch_size, seq_len, seq_len]
            attention_mask = torch.gt((attention_mask + subsequence_mask), 0)
        else:
            attention_mask = subsequence_mask.bool()
        # 计算每一层的结果
        self_attns = []
        for layer in self.layers:
            # outputs: [batch_size, seq_len, d_model],
            # self_attn: [batch_size, n_heads, seq_len, seq_len],
            outputs, self_attn = layer(outputs, attention_mask)
            self_attns.append(self_attn)
        return outputs, self_attns
  
  def get_attn_subsequence_mask(self,seq,device):
    #生成上三角掩码
    ## 注意力分数的大小是 [batch_size, n_heads, len_seq, len_seq]
    #所以这里要生成 [batch_size, len_seq, len_seq] 大小
    attn_shape=[seq.size(0),seq.size(1),seq.size(1)]
    #triu获得矩阵上三角部分
    subsequence_mask=np.triu(np.ones(attn_shape),k=1).astype(np.uint8)
    subsequence_mask = torch.tensor(subsequence_mask, dtype=torch.bool).to(device)
    return subsequence_mask
  
  def get_attn_pad_mask(self,attention_mask):
    #attention_mask 的掩码大小调整，要转换成 [batch_size, len_seq, len_seq] 大小，方便和注意力分数计算
    batch_size,len_seq=attention_mask.size()
    #eq返回布尔张量，逐元素比较，=0返回True
    attention_mask=attention_mask.data.eq(0).unsqueeze(1)
    #[batch_size, len_seq, len_seq]大小
    return attention_mask.expand(batch_size,len_seq,len_seq)



'''上面构建好解码器之后，就可以得到处理后的特征，下面还需要将特征转为词表大小的概率分布，才能实现对下一个Token的预测。'''
class GPTModel(nn.Module):
  def __init__(self,d_model,n_heads,d_ff,d_k,d_v,vocab_size,max_pos,n_layers,device):
    super(GPTModel,self).__init__()
    #解码器
    self.decoder=Decoder(d_model,n_heads,d_ff,d_k,d_v,vocab_size,max_pos,n_layers,device)
    #映射为词表大小
    self.projection=nn.Linear(d_model,vocab_size)

  def forward(self,inputs,attention_mask=None):
    outputs,self_attns=self.decoder(inputs,attention_mask)
    #这是模型对每个位置的预测得分:[batch_size, seq_len, vocab_size]
    logits=self.projection(outputs)
    #展开为[batch_size*seq_len, vocab_size]
    return logits.view(-1,logits.size(-1)),self_attns