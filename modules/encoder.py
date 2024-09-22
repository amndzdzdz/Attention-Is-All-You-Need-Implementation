import torch
import torch.nn as nn
from modules.attention import MultiHeadAttention
from modules.feed_forward import FeedForward
from modules.add_and_norm import AddAndNorm

class Encoder(nn.Module):
    """
    The Encoder only processes only the source sentence and produces the attention vectors 
    """
    def __init__(self, d_model):
        super(Encoder, self).__init__()
        self.multiHeadAttention = MultiHeadAttention(n_heads=8, d_model=d_model, mask=False)
        self.addAndNorm = AddAndNorm(d_model)
        self.feedFoward = FeedForward(d_model=d_model, d_ff=2048)

    def forward(self, x):
        identity1 = x
        x = self.multiHeadAttention(x, x, x)
        x = self.addAndNorm(identity1, x)

        identity2 = x
        x = self.feedFoward(x)
        x = self.addAndNorm(identity2, x)

        return x

if __name__ == '__main__':
    dummy_embedding = torch.rand(2, 10, 512)
    d_model = dummy_embedding.shape[-1]
    encoder = Encoder(d_model=d_model)
    encoder(dummy_embedding)
