import torch.nn as nn
import torch
from modules.encoder import Encoder
from modules.decoder import Decoder
from modules.embedding import EmbeddingLayer
from utils import create_dataloaders

class Transformer(nn.Module):
    def __init__(self, d_model, n_blocks, n_classes, vocab_size):
        super(Transformer, self).__init__()
        self.embedding = EmbeddingLayer(vocab_size=vocab_size, d_model=d_model)
        self.encoder = Encoder(d_model=d_model)
        self.decoder = Decoder(d_model=d_model)
        self.n_blocks = n_blocks
        self.ln = nn.Linear(32, vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, source, target, train=True):
        x = self.embedding(source)
        
        for _ in range(self.n_blocks):
            x = self.encoder(x)
        encoder_output = x

        if train:
            decoder_input = self.embedding(target)
        else:
            #if train=False (inference) give the decoder the start token "[<s>]"
            #TODO Add start token to a batch of decoder_inputs
            #Autoregressively predict the next token
            decoder_input = self.embedding(torch.tensor(0)) 

        x = self.decoder(decoder_input, encoder_output, encoder_output)
        for _ in range(self.n_blocks - 1):
            x = self.decoder(x, encoder_output, encoder_output)
        
        x = self.ln(x)

        return x