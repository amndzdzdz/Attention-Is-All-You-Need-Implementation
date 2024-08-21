import torch.nn as nn

class decoder_block(nn.Module):
    def __init__(self):
        super(decoder_block, self).__init__()

    def forward(self, x):
        return x