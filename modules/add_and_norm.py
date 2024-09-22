import torch.nn as nn
from modules.feed_forward import FeedForward

class AddAndNorm(nn.Module):
    """
    Skip Connection Block + Layer Normalization
    """
    def __init__(self, dim):
        super(AddAndNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, identity, x):
        out = identity + x
        out = self.norm(out)
        return out