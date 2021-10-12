import pandas as pd

def split_sequence(sequence, n):
    a, b, c = zip(*[iter(sequence)]*n), zip(*[iter(sequence[1:])]*n), zip(*[iter(sequence[2:])]*n)
    seq_ngrams = []
    for ngrams in [a, b, c]:
        temp = []
        for ngram in ngrams:
            temp.append("".join(ngram))
        seq_ngrams.append(temp)
    return seq_ngrams


def create_corpus(file, n, corpus_file):
    df = pd.read_csv(file)
    sub_seqs = df['sequence']
    with open(corpus_file, "w") as f:
        for sub_seq in sub_seqs:
            # print(sub_seq)
            ngrams = split_sequence(sub_seq, n)
            for ngram in ngrams:
                f.write(" ".join(ngram) + "\n")
