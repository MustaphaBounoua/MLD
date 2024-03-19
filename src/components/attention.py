
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        # self.norm = nn.BatchNorm1d( dim)
        # self.norm = nn.LayerNorm(dim, eps=1e-6 )
        self.norm = nn.GroupNorm(32, dim)
    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, num_heads = 4 ,dim_head = None):
        super().__init__()
        
     
        if dim_head == None:
            self.dim_head = input_dim//num_heads
        else: 
            self.dim_head = dim_head
            
        self.embed_dim = self.dim_head * num_heads
        self.num_heads = num_heads
        
        
        
        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional

        self.qkv_proj = nn.Linear( input_dim, self.embed_dim * 3,bias = False)
        
        self.o_proj = nn.Linear(self.embed_dim, input_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
       # self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length = x.size()
        
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size,  self.num_heads, 1, 3 * self.dim_head)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)
       # values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, self.embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        else:
            return o