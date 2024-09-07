import torch
import torch.nn as nn
from attention import MultiHeadAttention
from encoder import Encoder
from feed_forward import FeedForward
from add_and_norm import AddAndNorm

class Decoder(nn.Module):
    def __init__(self, d_embedding):
        super(Decoder, self).__init__()
        self.maskedMultiHeadAttention = MultiHeadAttention(n_heads=8, d_embedding=d_embedding, mask=True)

        self.multiHeadAttention = MultiHeadAttention(n_heads=8, d_embedding=d_embedding, mask=False)
        
        self.addAndNorm = AddAndNorm(d_embedding)

        self.feedFoward = FeedForward(d_model=d_embedding, d_ff=2048)

    def forward(self, x, query, key):
        identity1 = x
        x = self.maskedMultiHeadAttention(x, x, x)
        x = self.addAndNorm(identity1, x)

        identity2 = x
        x = self.multiHeadAttention(query, key, x)
        x = self.addAndNorm(identity2, x)

        identity3 = x
        x = self.feedFoward(x)
        x = self.addAndNorm(identity3, x)

        return x

if __name__ == '__main__':
    dummy_embedding = torch.rand(2, 10, 512)
    d_embedding = dummy_embedding.shape[-1]
    
    encoder = Encoder(d_embedding=d_embedding)
    decoder = Decoder(d_embedding=d_embedding)
    out_enc = encoder(dummy_embedding)

    out_dec = decoder(dummy_embedding, out_enc, out_enc)