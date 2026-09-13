import  torch
import torch.nn as nn
from .config import Config

# Linear class
class Linear(nn.Module):
    def __init__(self,d_in,d_out):
        super().__init__()
        self.sigma=((1/(d_in+d_out))*2)**0.5
        self.d_in=d_in
        self.d_out=d_out
        self.weight=nn.Parameter(nn.init.trunc_normal_(torch.empty(d_out,d_in),std=self.sigma,a=-3*self.sigma,b=3*self.sigma))

    def forward(self,in_features):
        assert self.d_in==in_features.size(-1)
        return in_features@self.weight.T

# Embedding,it will be used for token_id convert to hidden dimension
class Embedding(nn.Module):
    def __init__(self,vocab_size,d_model):
        super().__init__()
        self.weight=nn.Parameter(nn.init.trunc_normal_(torch.empty(vocab_size,d_model),a=-3,b=3))

    def forward(self,token_ids):
        return self.weight[token_ids]


class Swiglu(nn.Module):
    def __init__(self,d_model,d_ff):
        super().__init__()
        self.w1=Linear(d_model,d_ff)
        self.w2=Linear(d_ff,d_model)
        self.w3=Linear(d_model,d_ff)
        self.d_model=d_model
        self.d_ff=d_ff

    def forward(self,in_features):
        assert self.d_model==in_features.size(-1)
        return self.w2(silu(self.w1(in_features))*self.w3(in_features))


def silu(in_features):
    sigmoid=1.0/(1+torch.exp(-1.0*in_features))
    return in_features*sigmoid


def scaled_dot_product_attention(Q,K,V,mask=None):
    d_k=K.size(-1)
    scaled=d_k**-0.5
    attn_map=Q@K.transpose(-1,-2)*scaled
    if mask is not None:
        attn_map=attn_map.masked_fill(~mask,float('-inf'))
    attn_map=softmax(attn_map,dim=-1)
    return attn_map@V



def softmax(in_features,dim=-1):
    max_value,_=torch.max(in_features,dim=dim,keepdim=True)
    in_features=in_features-max_value
    exp_value=torch.exp(in_features)
    sum_value=torch.sum(exp_value,dim=dim,keepdim=True)
    return exp_value/sum_value


class CausalAttention(nn.Module):
    def __init__(self,config:Config,use_rope=False):
        super().__init__()
        assert config.d_model%config.num_heads==0
        self.config=config
        self.q_proj=Linear(config.d_model,config.d_model)
        self.k_proj=Linear(config.d_model,config.d_model)
        self.v_proj=Linear(config.d_model,config.d_model)
        self.output_proj=Linear(config.d_model,config.d_model)
        self.d_k=config.d_model//config.num_heads
        self.use_rope=use_rope
        if config.max_seq_len is not None and config.theta is not None and use_rope:
            self.rope=RotaryPositionalEmbedding(config.theta,self.d_k,config.max_seq_len)

    def forward(self,in_features,token_positions=None):
        *batch_size,seq_len,_=in_features.shape

        d_k=self.config.d_model//self.config.num_heads

        mask=torch.tril(torch.ones(seq_len,seq_len,dtype=torch.bool,device=in_features.device))

        # shape (d_model,3*d_model)
        weights=torch.cat([self.q_proj.weight.T,self.k_proj.weight.T,self.v_proj.weight.T],dim=-1)

        # shape (...,seq_len,3*d_model)
        qkv=in_features@weights

        Q,K,V=qkv.split(self.config.d_model,dim=-1)

        # shape (batch_size,num_heads,seq_len,d_k)
        Q=Q.contiguous().view(*batch_size,seq_len,self.config.num_heads,self.d_k).transpose(-2,-3)
        K=K.contiguous().view(*batch_size,seq_len,self.config.num_heads,self.d_k).transpose(-2,-3)
        V=V.contiguous().view(*batch_size,seq_len,self.config.num_heads,self.d_k).transpose(-2,-3)

        # 是否使用rope
        if self.use_rope:
                    if token_positions is None:
                        token_positions=torch.arange(seq_len,device=in_features.device)
                    token_positions=token_positions.unsqueeze(-2)
                    Q=self.rope(Q,token_positions)
                    K=self.rope(K,token_positions)

        # shape (batch_size,num_heads,seq_len,d_k)
        y=scaled_dot_product_attention(Q,K,V,mask)

        y=y.transpose(-2,-3).contiguous().view(*batch_size,seq_len,self.config.d_model)

        return self.output_proj(y)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self,theta,d_k,max_seq_len,device=None):
        super().__init__()
        assert d_k%2==0
        self.d_k=d_k
        self.theta=theta
        self.max_seq_len=max_seq_len
        self.device=device
        # shape (d_k//2,)
        self.freq_v=torch.pow(theta,-1.0*torch.arange(0,d_k,2,dtype=torch.float32,device=device)/d_k)
        # shape (max_seq_len,d_k//2)
        self.rotate_angle=torch.outer(torch.arange(max_seq_len,dtype=torch.float32,device=device),self.freq_v)
        self.register_buffer("cos_table",torch.cos(self.rotate_angle),persistent=False)
        self.register_buffer("sin_table",torch.sin(self.rotate_angle),persistent=False)


    def forward(self,in_query_or_key,token_positions):
        # shape (seq_len,k//2)
        real_cos_table=self.cos_table[token_positions]
        real_sin_table=self.sin_table[token_positions]

        *batch_dim,seq_len,d_k=in_query_or_key.shape

        in_query_or_key_pair=in_query_or_key.contiguous().view(*batch_dim,seq_len,d_k//2,2)
        
        emb_query_or_key=torch.stack([in_query_or_key_pair[...,0]*real_cos_table-in_query_or_key_pair[...,1]*real_sin_table,in_query_or_key_pair[...,0]*real_sin_table+in_query_or_key_pair[...,1]*real_cos_table],dim=-1)

        emb_query_or_key=emb_query_or_key.reshape(*batch_dim,seq_len,d_k)

        return emb_query_or_key


class RMSNorm(nn.Module):
    def __init__(self,d_model,eps):
        super().__init__()
        self.d_model=d_model
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(d_model))


    def forward(self,in_features):
        assert self.d_model==in_features.size(-1)
        # shape (...,d_model)
        rms=torch.rsqrt(torch.mean(in_features*in_features,dim=-1,keepdim=True)+self.eps)
        in_features=in_features*rms
        return in_features*self.weight

class TransformerBlock(nn.Module):
    def __init__(self,config:Config):
        super().__init__()
        self.config=config
        self.ffn=Swiglu(config.d_model,config.d_ff)
        self.attn=CausalAttention(config,use_rope=True)
        self.ln1=RMSNorm(config.d_model,config.eps)
        self.ln2=RMSNorm(config.d_model,config.eps)

    def forward(self,in_features,token_positions=None):
        y1=in_features+self.attn(self.ln1(in_features),token_positions)
        y2=y1+self.ffn(self.ln2(y1))
        return y2

    
class Transformer_LM(nn.Module):
    def __init__(self,config:Config):
        super().__init__()
        self.layers=nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.token_embeddings=Embedding(config.vocab_size,config.d_model)
        self.ln_final=RMSNorm(config.d_model,config.eps)
        self.lm_head=Linear(config.d_model,config.vocab_size)

    def forward(self,in_indices,token_positions=None):
        ids=self.token_embeddings(in_indices)
        for layer in self.layers:
            ids=layer(ids,token_positions)
        
        return self.lm_head(self.ln_final(ids))




              
        

        

    
