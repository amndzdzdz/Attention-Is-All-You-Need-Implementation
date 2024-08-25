import torch.nn as nn
from feed_forward import FeedForward

class FeedForwardBlock(nn.Module):
    def __init__(self, x, d_model, d_ff):
        super(FeedForwardBlock, self).__init__()
        self.norm = nn.LayerNorm()
        self.feedforward = FeedForward(d_model, d_ff)

    def forward(self, x):
        identity = x
        x = self.feedforward(x)
        x = identity + x
        x = self.norm(x)
        return x