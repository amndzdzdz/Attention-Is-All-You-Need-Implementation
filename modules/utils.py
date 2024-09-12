from nltk.tokenize import word_tokenize

def preprocess_input(input_sequence, vocab_dict):
    tokenized_input = word_tokenize(input_sequence)
    tokenized_input = [word.lower() for word in tokenized_input]
    output = []
    for token in tokenized_input:
        output.append(vocab_dict[token])

    return torch.LongTensor(output)

def create_vocabularies(filepath):
    en_vocabulary = {}
    ar_vocabulary = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            en, ar = line.replace("\n", "").split('\t')
            
            if en in en_vocabulary.keys():
                en_vocabulary[en] += 1
            else:
                en_vocabulary[en] = 1
            
            if ar in ar_vocabulary.keys():
                ar_vocabulary[ar] += 1
            else:
                ar_vocabulary[ar] = 1
    
    return en_vocabulary, ar_vocabulary


if __name__ == '__main__':

    vocab_dict = {
        "this": 0,
        "is": 1,
        "a": 2,
        "test": 3,
        "sequence": 4
    }

    input_sequence = "This is a test sequence"
    preprocessed_input = preprocess_input(input_sequence=input_sequence, voacb_dict=vocab_dict)
    print(preprocessed_input)
