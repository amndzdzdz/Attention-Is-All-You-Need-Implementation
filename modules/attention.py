import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ScaledDotProductAttention(nn.Module):
    def __init__(self, dk, mask=False):
        super(ScaledDotProductAttention, self).__init__()
        self.dk = dk
        self.mask = mask

    def forward(self, query, keys, values):
        #Calculate the dot product of the query and the transposed key
        q_k_matmul = torch.matmul(query, torch.transpose(keys, dim0=2, dim1=3))
        
        #Mask the future tokens (tokens above the diagonal) (relevant for the decoder)
        if self.mask:
            inf_matrix = torch.ones(q_k_matmul.shape) * float('-inf')
            mask_matrix = torch.triu(inf_matrix, diagonal=1)
            q_k_matmul = q_k_matmul + mask_matrix

        #scale the dot product with the inverse of the key dimension
        q_k_matmul = torch.div(q_k_matmul, math.sqrt(self.dk))

        #q_k_matmul = q_k_matmul.apply_(lambda x: x / torch.sqrt(torch.tensor(self.dk)))
        q_k_softmax = F.softmax(q_k_matmul, dim=-1)
        x = torch.matmul(q_k_softmax, values)
        return x

if __name__ == '__main__':
    query = torch.rand(1, 2, 32, 16, device="cpu")
    key = torch.rand(1, 2, 32, 16, device="cpu")
    value = torch.rand(1, 2, 32, 16, device="cpu")
    key_dim = query.shape[-1]
    value_dim = value.shape[-1]
    x = dot_prod(query, key, value)
    print(x)