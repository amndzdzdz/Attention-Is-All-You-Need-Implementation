import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ScaledDotProductAttention(nn.Module):
    """
    Implementation of the Scaled Dot Product Attention from the 
    "Attention Is All You Need Paper"
    """
    def __init__(self, dim_k, mask=False):
        super(ScaledDotProductAttention, self).__init__()
        self.dim_k = dim_k
        self.mask = mask

    def forward(self, query, keys, values):
        #Calculate the dot product of the query and the transposed key
        q_k_matmul = torch.matmul(query, torch.transpose(keys, dim0=2, dim1=3))
        
        #Mask the future tokens (tokens above the diagonal) (relevant for the decoder)
        if self.mask:
            inf_matrix = torch.ones(q_k_matmul.shape) * -1e9
            mask_matrix = torch.triu(inf_matrix, diagonal=1)
            q_k_matmul = q_k_matmul + mask_matrix

        #scale the dot product with the inverse of the key dimension
        q_k_matmul = torch.div(q_k_matmul, math.sqrt(self.dim_k))

        q_k_softmax = F.softmax(q_k_matmul, dim=-1)
        x = torch.matmul(q_k_softmax, values)
        return x


class MultiHeadAttention(nn.Module):
    """
    Implementation of the (Masked) Multi Head Attention Module from the
    "Attention Is All You Need" Paper
    """
    def __init__(self, n_heads, d_model, mask):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.dims_at_hand = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(d_model, mask=mask)

        self.ln4 = nn.Linear(d_model, d_model)

    def forward(self, query, key, value):
        #Linearly Project the q, k and v first
        query = self.q_linear(query)
        key = self.k_linear(key)
        value = self.v_linear(value)

        seq_len = query.shape[1]
        batch_size = query.shape[0]

        #Split tensors across the n_heads and change place of seq_len and num_heads
        query = query.view(batch_size, -1, self.n_heads, self.dims_at_hand).transpose(1, 2)
        key = key.view(batch_size, -1, self.n_heads, self.dims_at_hand).transpose(1, 2)
        value = value.view(batch_size, -1, self.n_heads, self.dims_at_hand).transpose(1, 2)

        #calculate attention scores
        attention_scores = self.attention(query, key, value)

        #concat the attention scores across the n_heads 
        attention_scores = attention_scores.transpose(-1, -2).contiguous().view(batch_size, -1, self.d_model)

        attention_scores = self.ln4(attention_scores)
        
        return attention_scores

if __name__ == '__main__':
    query = torch.rand(1, 10, 512, device="cpu")
    key = torch.rand(1, 10, 512, device="cpu")
    value = torch.rand(1, 10, 512, device="cpu")
    dim_q = query.shape[-1]
    dim_v = value.shape[-1]
    attention = MultiHeadAttention(8, dim_q, mask=True)
    x = attention(query, key, value)
    print(x)