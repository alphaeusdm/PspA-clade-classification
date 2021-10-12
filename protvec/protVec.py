from gensim.models import word2vec, KeyedVectors
from protvec.utils import *

class ProtVec(word2vec.Word2Vec):
    def __init__(self, seq_file=None, corpus_file='corpus.txt', corpus=None, n=3, size=100, sg=1, window=25, workers=1, min_count=1):
        self.n = n
        # self.size = size
        self.seq_file = seq_file

        if seq_file is None and corpus is None:
            raise Exception("Sequence file or corpus required.")

        if seq_file is not None:
            create_corpus(seq_file, n, corpus_file)
            corpus = word2vec.Text8Corpus(corpus_file)

        word2vec.Word2Vec.__init__(self, corpus, sg=sg, window=window, min_count=min_count, workers=workers)

    def seq_to_vec(self, sequence):
        ngrams = split_sequence(sequence, self.n)
        prot_vec = []
        for ngram in ngrams:
            ngram_vec = []
            for subseq in ngram:
                try:
                    ngram_vec.append(self[subseq])
                except:
                    raise ("Model not trained")
            prot_vec.append(sum(ngram_vec))
        return prot_vec