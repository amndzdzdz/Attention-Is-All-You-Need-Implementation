import torch
import torch.nn as nn
from feed_forward import FeedForward
from utils import preprocess_input

class EmbeddingLayer(nn.Module):
    """
    Embedding-Layer + Positional Encodings
    """
    def __init__(self, vocab_size, d_model):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model) #vocab_size = columns, embedding_dim = rows
        self.positionalEncodings = self._make_positional_encodings(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding(x) + self.positionalEncodings

    #TODO make this way more efficiant
    def _make_positional_encodings(self, vocab_size, d_model):
        embeddings = torch.zeros((vocab_size, d_model))
        for pos in range(vocab_size):
            for i in range(d_model):
                if i % 2 == 0:
                    embeddings[pos, i] = torch.sin(torch.tensor(pos / 10000 ** (2*i / d_model)))
                else: 
                    embeddings[pos, i] = torch.cos(torch.tensor(pos / 10000 ** (2*i / d_model)))
                
        return embeddings


if __name__ == '__main__':
    input_sequence = "This is a test sequence"
    vocab_dict = {
        "this": 0,
        "is": 1,
        "a": 2,
        "test": 3,
        "sequence": 4
    }
    preprocessed_input = preprocess_input(input_sequence=input_sequence, vocab_dict=vocab_dict)

    embeddingLayer = EmbeddingLayer(5, 512)
    embeddingLayer(preprocessed_input)



