from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace, WhitespaceSplit
from tokenizers.trainers import WordPieceTrainer
from tokenizers.models import WordPiece
from tokenizers.processors import TemplateProcessing

def train_tokenizer(seq_len):
    train_data = load_dataset('wmt14','de-en',split='train').flatten()
    test_data = load_dataset('wmt14','de-en',split='test').flatten()
    val_data = load_dataset('wmt14','de-en',split='validation').flatten()
    
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = WhitespaceSplit()

    def data_iterator(train_data, test_data, val_data):
        for data_set in [val_data]:
            for row in data_set:
                yield row["translation.de"]
            for row in data_set:
                yield row["translation.en"]

    trainer = WordPieceTrainer(special_tokens=["[<s>]", "[</s>]", "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train_from_iterator(data_iterator(train_data, test_data, val_data), trainer)
    tokenizer.post_processor = TemplateProcessing(
    single="[<s>] $A [</s>]",
    special_tokens=[
        ("[<s>]", 2),
        ("[</s>]", 3),
    ])
    tokenizer.enable_padding(pad_id=4, pad_token="[PAD]", length=seq_len)
    tokenizer.enable_truncation(max_length=seq_len)
    tokenizer.save("tokenizer_checkpoint/tokenizer_checkpoint.json")

def load_tokenizer(check_point):
    return Tokenizer.from_file(check_point)

if __name__ == '__main__':
    train_tokenizer(5)