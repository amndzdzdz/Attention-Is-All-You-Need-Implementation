import torch.nn as nn

class transformerModel(nn.Module):
    def __init__(self):
        super(transformerModel).__init__()

    def forward(self, x):
        return x