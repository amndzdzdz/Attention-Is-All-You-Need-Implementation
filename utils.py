

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
    en_vocabulary, ar_vocabulary = create_vocabularies("data/ara_eng.txt")