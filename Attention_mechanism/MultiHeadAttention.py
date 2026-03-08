import torch
import torch.nn as nn 

class MultiHeadAttention(nn.Module):

    def __init__(self, d_in, d_out, seq_len, dropout, num_heads, qkv_bias = False):
        super().__init__()

        assert (d_out % num_heads == 0), "d_out has to be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.d_k = d_out // num_heads

        self.W_k = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_q = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_v = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(seq_len, seq_len), diagonal=1))


    def forward(self, x):
        b, seq_len, d_in = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x) #Tensor shape: (b, seq_len, d_out)

        #I now need to split d_out into the 'multiple heads' by using the view attribute
        Q = Q.view(b, seq_len, self.num_heads, self.d_k)
        K = K.view(b, seq_len, self.num_heads, self.d_k)
        V = V.view(b, seq_len, self.num_heads, self.d_k) #Tensor shape: (b, seq_len, num_heads, d_k)

        #Now I need to 'swap' the seq_len and num_heads dimensions. This is so the matrix multiplication works correctly 
        Q = Q.transpose(1,2)
        K = K.transpose(1,2)
        V = V.transpose(1,2)#Tensor shape: (b, num_heads, seq_len, d_k)

        #compute dot product for EACH head
        attn_scores = Q @ K.transpose(2,3)

        mask_bool = self.mask.bool()[:seq_len, :seq_len]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / K.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ V).transpose(1,2) #Tensor shape: (b, seq_len, num_heads, d_k)

        context_vec = context_vec.contiguous().view(b, seq_len, self.d_out)# Tensor shape: (b, seq_len, d_out)

        context_vec = self.out_proj(context_vec)

        return context_vec
