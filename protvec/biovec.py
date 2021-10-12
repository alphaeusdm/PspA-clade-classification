from protVec import *


model = ProtVec(seq_file='../proline_rich.csv')
model.wv.save_word2vec_format('vector_embeddings_proline_rich.txt', binary=False)


