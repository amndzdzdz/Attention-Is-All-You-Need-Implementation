import torch.nn as nn
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.embedding import EmbeddingLayer

class Transformer(nn.Module):
    def __init__(self, d_model, n_blocks, n_classes):
        super(Transformer).__init__()
        self.embedding = EmbeddingLayer(vocab_size=1000, d_model=d_model)
        self.encoder = Encoder(d_model=d_model)
        self.decoder = Decoder(d_model=d_model)
        
        self.start_token = self._create_start_token()
        self.end_token = self._create_end_token()
        self.ln = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.embedding(x)

        for _ in range(n_blocks):
            x = self.encoder(x)
        encoder_output = x

        x = self.decoder(self.start_token, encoder_output, encoder_output)
        for _ in range(n_blocks - 1):
            x = self.decoder(x, encoder_output, encoder_output)

        x = torch.flatten(x)

        return x

    def _create_start_token(self):
        return None

    def _create_end_token(self):
        return None