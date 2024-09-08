from nltk.tokenize import word_tokenize

def preprocess_input(input_sequence, vocab_dict):
    tokenized_input = word_tokenize(input_sequence)
    tokenized_input = [word.lower() for word in tokenized_input]
    output = []
    for token in tokenized_input:
        output.append(vocab_dict[token])

    return torch.LongTensor(output)


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
