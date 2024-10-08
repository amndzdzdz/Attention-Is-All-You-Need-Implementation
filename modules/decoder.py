import torch
import torch.nn as nn
from modules.attention import MultiHeadAttention
from modules.encoder import Encoder
from modules.feed_forward import FeedForward
from modules.add_and_norm import AddAndNorm

class Decoder(nn.Module):
    def __init__(self, d_model):
        super(Decoder, self).__init__()
        self.maskedMultiHeadAttention = MultiHeadAttention(n_heads=8, d_model=d_model, mask=True)

        self.multiHeadAttention = MultiHeadAttention(n_heads=8, d_model=d_model, mask=False)
        
        self.addAndNorm = AddAndNorm(d_model)

        self.feedFoward = FeedForward(d_model=d_model, d_ff=2048)

    def forward(self, x, key, value): #x is the previous Decoder Output, key and value come from the encoder
        identity1 = x
        x = self.maskedMultiHeadAttention(x, x, x)
        x = self.addAndNorm(identity1, x)

        identity2 = x
        x = self.multiHeadAttention(x, key, value)
        x = self.addAndNorm(identity2, x)

        identity3 = x
        x = self.feedFoward(x)
        x = self.addAndNorm(identity3, x)

        return x

if __name__ == '__main__':
    dummy_embedding = torch.rand(2, 10, 512)
    d_model = dummy_embedding.shape[-1]
    
    encoder = Encoder(d_model=d_model)
    decoder = Decoder(d_model=d_model)
    out_enc = encoder(dummy_embedding)

    out_dec = decoder(dummy_embedding, out_enc, out_enc)