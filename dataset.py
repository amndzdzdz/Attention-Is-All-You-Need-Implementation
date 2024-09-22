import torch.nn as nn
from torch.utils.data import Dataset

class TranslationDataset(Dataset):
    def __init__(self, data):
        self.data = data 
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        src_sentence = sample["translation.de"]
        trgt_sentence = sample["translation.en"]
        return src_sentence, trgt_sentence

    def __len__(self):
        return len(self.data)