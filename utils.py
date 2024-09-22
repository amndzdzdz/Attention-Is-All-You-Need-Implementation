from torch.utils.data import Dataset, DataLoader
from tokenizer import load_tokenizer
from datasets import load_dataset
import torch

class TranslationDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    
    def __getitem__(self, idx):
        sample = self.data[idx]

        src_sentence = self.tokenizer.encode(sample["translation.de"]).ids
        trgt_sentence = self.tokenizer.encode(sample["translation.en"]).ids
        
        return torch.tensor(src_sentence), torch.tensor(trgt_sentence)

    def __len__(self):
        return len(self.data)

def create_dataloaders(tokenizer_checkpoint):
    train_data = load_dataset('wmt14','de-en',split='train').flatten()
    test_data = load_dataset('wmt14','de-en',split='test').flatten()
    val_data = load_dataset('wmt14','de-en',split='validation').flatten()

    tokenizer = load_tokenizer(tokenizer_checkpoint)

    train_dataset = TranslationDataset(train_data, tokenizer)
    val_dataset = TranslationDataset(val_data, tokenizer)
    test_dataset = TranslationDataset(test_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    train_loader, val_loader, test_loader = create_dataloaders("tokenizer_checkpoint/tokenizer_checkpoint.json")

    for batch in train_loader:
        src_sentence, trgt_sentence = batch