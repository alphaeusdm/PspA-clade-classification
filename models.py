import math
import warnings

# import prettytable as PrettyTable
from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
warnings.filterwarnings("ignore")

embeddings = {}
vectors = pd.read_csv('protvec/vector_embeddings_all.txt', sep=' ', skiprows=1, header=None).values
for i, vec in enumerate(vectors):
    embeddings[vec[0]] = np.array(vec[1:])


def seq_to_vectors(data):
    # load embeddings from the vector file
    vectors = pd.read_csv('protvec/vector_embeddings_all.txt', sep=' ', skiprows=1, header=None).values
    for i, vec in enumerate(vectors):
        embeddings[vec[0]] = np.array(vec[1:])

    # convert sequences to vectors
    df = pd.DataFrame(columns=['seq_emb', 'clade'])
    for ind in data.index:
        sequence = data['sequence'][ind]
        seq = " ".join(sequence)
        tokens = [token for token in seq.split(" ") if token != ""]
        ngrams = zip(*[tokens[i:] for i in range(3)])
        ngrams = ["".join(ngram) for ngram in ngrams]
        embs = []
        for ngram in ngrams:
            # print(ngram, embeddings.get(ngram))
            if ngram in embeddings.keys():
                emb = embeddings[ngram]
                if len(emb) > 0:
                    embs.append(emb)
            else:
                continue
        df = df.append({'seq_emb':np.vstack(embs), 'clade':data['clade'][ind]}, ignore_index=True)

    return df

def seq_to_vec(sequence):
    # convert sequence to vector
    seq = " ".join(sequence)
    tokens = [token for token in seq.split(" ") if token != ""]
    ngrams = zip(*[tokens[i:] for i in range(3)])
    ngrams = ["".join(ngram) for ngram in ngrams]
    print(ngrams)
    embs = []
    for ngram in ngrams:
        # print(ngram, embeddings.get(ngram))
        if ngram in embeddings.keys():
            emb = embeddings[ngram]
            if len(emb) > 0:
                embs.append(emb)
        else:
            continue
    return embs
    # return np.vstack(embs)



mean_vectors_train = []
norms_mean_vectors_train = []
covariance_matrices_train = []
norms_covariance_matrices_train = []
def build_kernal_matrix(train_matrices, test_matrices, k, alpha, run_test=False):
    # build kernel matrices(feature matrices).
    # based on the paper 'Machine learning predicts nucleosome binding modes of transcription factors'.

    # mean_vectors_train = []
    # norms_mean_vectors_train = []
    # covariance_matrices_train = []
    # norms_covariance_matrices_train = []
    for matrix in train_matrices:
        if matrix.shape[0] == 1:
            M = np.sum(matrix, axis=0) / float(matrix.shape[0])
            C = np.outer(M, M).flatten()
            mean_vectors_train.append(M)
            norms_mean_vectors_train.append(np.linalg.norm(M))
            covariance_matrices_train.append(C)
            norms_covariance_matrices_train.append(np.linalg.norm(C))
        elif matrix.shape[0] > 0:
            M = np.sum(matrix, axis=0) / float(matrix.shape[0])
            C = np.cov(matrix, rowvar=False).flatten()
            mean_vectors_train.append(M)
            norms_mean_vectors_train.append(np.linalg.norm(M))
            covariance_matrices_train.append(C)
            norms_covariance_matrices_train.append(np.linalg.norm(C))
        else:
            M = np.zeros(k)
            C = np.zeros((k, k)).flatten()
            mean_vectors_train.append(M)
            norms_mean_vectors_train.append(1)
            covariance_matrices_train.append(C)
            norms_covariance_matrices_train.append(1)

    kernel_matrix_train = np.zeros((len(train_matrices), len(train_matrices)), dtype=np.float16)
    for i in range(len(train_matrices)):
        for j in range(i, len(train_matrices)):
            kernel_matrix_train[i, j] = alpha * (np.dot(mean_vectors_train[i], mean_vectors_train[j]) / (
                        norms_mean_vectors_train[i] * norms_mean_vectors_train[j])) + (1 - alpha) * (
                                                    np.dot(covariance_matrices_train[i],
                                                           covariance_matrices_train[j]) / (
                                                                norms_covariance_matrices_train[i] *
                                                                norms_covariance_matrices_train[j]))
            kernel_matrix_train[j, i] = kernel_matrix_train[i, j]

    kernel_matrix_test = np.zeros((len(test_matrices), len(train_matrices)))
    if run_test:
        mean_vectors_test = []
        norms_mean_vectors_test = []
        covariance_matrices_test = []
        norms_covariance_matrices_test = []
        for matrix in test_matrices:
            if matrix.shape[0] == 1:
                M = np.sum(matrix, axis=0) / float(matrix.shape[0])
                C = np.outer(M, M).flatten()
                mean_vectors_test.append(M)
                norms_mean_vectors_test.append(np.linalg.norm(M))
                covariance_matrices_test.append(C)
                norms_covariance_matrices_test.append(np.linalg.norm(C))
            elif matrix.shape[0] > 0:
                M = np.sum(matrix, axis=0) / float(matrix.shape[0])
                C = np.cov(matrix, rowvar=False).flatten()
                mean_vectors_test.append(M)
                norms_mean_vectors_test.append(np.linalg.norm(M))
                covariance_matrices_test.append(C)
                norms_covariance_matrices_test.append(np.linalg.norm(C))
            else:
                M = np.zeros(k)
                C = np.zeros((k, k)).flatten()
                mean_vectors_test.append(M)
                norms_mean_vectors_test.append(1)
                covariance_matrices_test.append(C)
                norms_covariance_matrices_test.append(1)

        for i in range(len(test_matrices)):
            for j in range(len(train_matrices)):
                kernel_matrix_test[i, j] = alpha * (np.dot(mean_vectors_test[i], mean_vectors_train[j]) / (
                            norms_mean_vectors_test[i] * norms_mean_vectors_train[j])) + (1 - alpha) * (
                                                       np.dot(covariance_matrices_test[i],
                                                              covariance_matrices_train[j]) / (
                                                                   norms_covariance_matrices_test[i] *
                                                                   norms_covariance_matrices_train[j]))
    return kernel_matrix_train, kernel_matrix_test


def build_matrix(seq, k, alpha):
    # build kernel matrix for new sequence
    mean_vectors = []
    norms_mean_vectors = []
    covariance_matrices = []
    norms_covariance_matrices = []
    kernel_matrix = np.zeros((1, train_len))
    seq = seq.astype(float)
    if seq.shape[0] == 1:
        M = np.sum(seq, axis=0) / float(seq.shape[0])
        C = np.outer(M, M).flatten()
        mean_vectors.append(M)
        norms_mean_vectors.append(np.linalg.norm(M))
        covariance_matrices.append(C)
        norms_covariance_matrices.append(np.linalg.norm(C))
    elif seq.shape[0] > 0:
        M = np.sum(seq, axis=0) / float(seq.shape[0])
        C = np.cov(seq, rowvar=False).flatten()
        mean_vectors.append(M)
        norms_mean_vectors.append(np.linalg.norm(M))
        covariance_matrices.append(C)
        norms_covariance_matrices.append(np.linalg.norm(C))
    else:
        M = np.zeros(k)
        C = np.zeros((k, k)).flatten()
        mean_vectors.append(M)
        norms_mean_vectors.append(1)
        covariance_matrices.append(C)
        norms_covariance_matrices.append(1)

    for i in range(1):
        for j in range(train_len):
            kernel_matrix[i, j] = alpha * (np.dot(mean_vectors[i], mean_vectors_train[j]) / (
                    norms_mean_vectors[i] * norms_mean_vectors_train[j])) + (1 - alpha) * (
                                               np.dot(covariance_matrices[i],
                                                      covariance_matrices_train[j]) / (
                                                       norms_covariance_matrices[i] *
                                                       norms_covariance_matrices_train[j]))
    return kernel_matrix



def create_matrices(emb):
    # get the embeddings in matrix form
    seq_matrices = []
    for i, M in enumerate(emb):
        seq_matrices.append(M.astype(float))

    return seq_matrices

def ml_split(y, num_splits=10, seed=0):
    # perform cross validation
    kf = KFold(n_splits=num_splits, shuffle=True, random_state=seed)
    splits = []
    for t_idx, v_idx in kf.split(y):
        splits.append((t_idx, v_idx))

    return splits


def real_AUPR(label, score):
    # evaluation
    label = label.flatten()
    score = score.flatten()

    order = np.argsort(score)[::-1]
    label = label[order]

    P = np.count_nonzero(label)
    # N = len(label) - P

    TP = np.cumsum(label, dtype=float)
    PP = np.arange(1, len(label)+1, dtype=float)  # python

    np.seterr(invalid='ignore')
    x = np.divide(TP, P)  # recall
    y = np.divide(TP, PP)  # precision

    pr = np.trapz(y, x)
    f = np.divide(2*x*y, (x + y))
    idx = np.where((x + y) != 0)[0]
    if len(idx) != 0:
        f = np.max(f[idx])
    else:
        f = 0.0

    return pr, f


def evaluate_performance(y_test, y_score, y_pred, alpha):
    # evaluation
    n_classes = 6
    perf = dict()

    # Compute macro-averaged AUPR
    perf["M-aupr"] = 0.0
    n = 0
    for i in range(n_classes):
        perf[i], _ = real_AUPR(y_test[:, i], y_score[:, i])
        if sum(y_test[:, i]) > 0:
            n += 1
            perf["M-aupr"] += perf[i]
    if n == 0:
        n = 1
    perf["M-aupr"] /= n
    # Compute micro-averaged AUPR
    # pr, _ = real_AUPR(y_test, y_score)
    # perf["m-aupr"] = pr if pr == pr else 0

    # Computes accuracy
    perf['acc'] = accuracy_score(y_test, y_pred)

    # Computes F1-score
    perf["F1"] = f1_score(y_test, y_pred, average='micro')
    return perf

# read data file and get embeddings of sequences
data = pd.read_csv('data.csv', usecols=['sequence', 'clade'])
# data = data.loc[0:308]

# get number of clades
number_of_clades = 1
for clade in data['clade']:
    if clade > number_of_clades:
        number_of_clades = clade
data = seq_to_vectors(data)

# split data in train test set
train_set, test_set = train_test_split(data, shuffle=True, test_size=0.15, train_size=0.85)
x_train, y_train = train_set['seq_emb'], np.array(train_set['clade'].astype(int))
x_test, y_test = test_set['seq_emb'], np.array(test_set['clade'].astype(int))

train_len = len(x_train)

# convert labels to proper format for training if y =4 than [0,0,0,1,0,0]
y_temp = np.zeros((len(y_train), 6), dtype=int)
for index in range(len(y_train)):
    temp = y_temp[index]
    temp[y_train[index]-1] = 1
    y_temp[index] = temp
y_train = y_temp

y_temp = np.zeros((len(y_test), 6), dtype=int)
for index in range(len(y_test)):
    temp = y_temp[index]
    temp[y_test[index]-1] = 1
    y_temp[index] = temp
y_test = y_temp

classes = []
for i in range(1,number_of_clades+1):
    classes.append(i)

# select the configured binding modes
selected_classes = np.arange(6)
classes = list(map(classes.__getitem__, selected_classes))
y_train = y_train[:, selected_classes]
num_classes = len(classes)
y_train_temp = y_train

# for x in x_train:

x_train = create_matrices(x_train)
x_train_temp = x_train
x_test = create_matrices(x_test)

model = None

best_alpha = 0
max_micro_aupr = 0
max_acc = 0
c_penalty = 10

# start training here
for alpha in tqdm(range(0, 11, 1)):
    f_macro_aupr = []
    f_micro_aupr = []
    f_acc = []
    f_f1 = []

    X_train, X_test = build_kernal_matrix(x_train, x_test, 100, alpha/10)
    # X_train = x_tra
    y_labels = y_train
    for seed in range(10):
        macro_aupr = []
        micro_aupr = []
        acc = []
        f1 = []
        counter = 1
        splits = ml_split(y_labels, num_splits=10, seed=seed)

        for train, valid in splits:
            x_train = np.nan_to_num(X_train[train, :][:, train])
            x_valid = np.nan_to_num(X_train[valid, :][:, train])

            y_train = y_labels[train.astype(int)]
            y_val = y_labels[valid.astype(int)]

            model = RandomForestClassifier()
            # model = OneVsRestClassifier(svm.SVC(C=c_penalty, kernel='precomputed', random_state=42, probability=True), n_jobs=-1)
            model.fit(x_train, y_train)

            y_score_valid = model.predict_proba(x_valid)
            y_pred_valid = model.predict(x_valid)
            y_score_valid = np.array(y_score_valid)
            result = evaluate_performance(y_val, y_score_valid, np.array(y_pred_valid), 3)
            # result = evaluate_performance(y_val, np.array(y_pred_valid), 3)


            # micro_aupr.append(result['m-aupr'])
            macro_aupr.append(result['M-aupr'])
            f1.append(result['F1'])
            acc.append(result['acc'])
            counter += 1

        # f_micro_aupr.append(round(np.mean(micro_aupr), 3))
        f_macro_aupr.append(round(np.mean(macro_aupr), 3))
        f_f1.append(round(np.mean(f1), 3))
        f_acc.append(round(np.mean(acc), 3))

    # choose alpha based on macro-AUPR
    if max_micro_aupr <= round(np.mean(f_macro_aupr), 3) and max_acc <= round(np.mean(f_acc), 3):
        max_micro_aupr = round(np.mean(f_macro_aupr), 3)
        max_acc = round(np.mean(f_acc), 3)
        best_alpha = alpha / 10


x_train = x_train_temp
y_train = y_train_temp
x_train, x_test = build_kernal_matrix(x_train, x_test, 100, best_alpha, True)


model = RandomForestClassifier()
# model = OneVsRestClassifier(svm.SVC(C=c_penalty, kernel='precomputed', random_state=42, probability=True), n_jobs=-1)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print('accuracy: ', f1_score(y_test, y_pred, average='micro'))



