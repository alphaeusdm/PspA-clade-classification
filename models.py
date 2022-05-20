import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn import svm
import pickle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, plot_confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

number_of_clades = 1
embeddings = {}
def read_embeddings(filename):
    # 'protvec/vector_embeddings_all.txt'
    vectors = pd.read_csv(filename, sep=' ', skiprows=1, header=None).values
    global embeddings
    embeddings = {}
    for i, vec in enumerate(vectors):
        embeddings[vec[0]] = np.array(vec[1:])


def seq_to_vectors(data):
    # load embeddings from the vector file
    # vectors = pd.read_csv('protvec/vector_embeddings_all.txt', sep=' ', skiprows=1, header=None).values
    # for i, vec in enumerate(vectors):
    #     embeddings[vec[0]] = np.array(vec[1:])

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
    embs = []
    for ngram in ngrams:
        # print(ngram, embeddings.get(ngram))
        if ngram in embeddings.keys():
            emb = embeddings[ngram]
            if len(emb) > 0:
                embs.append(emb)
        else:
            continue
    # print(embs)
    return embs
    # return np.vstack(embs)


def build_kernal_matrix(train_matrices, test_matrices, k, alpha, run_test=False):
    # build kernel matrices(feature matrices).
    # based on the paper 'Machine learning predicts nucleosome binding modes of transcription factors'.

    vectors = {}
    mean_vectors_train = []
    norms_mean_vectors_train = []
    covariance_matrices_train = []
    norms_covariance_matrices_train = []
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

    vectors['mean_vectors_train'] = mean_vectors_train
    vectors['norms_mean_vectors_train'] = norms_mean_vectors_train
    vectors['covariance_matrices_train'] = covariance_matrices_train
    vectors['norms_covariance_matrices_train'] = norms_covariance_matrices_train
    return kernel_matrix_train, kernel_matrix_test, vectors


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
    N = len(label) - P

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

    return pr, f, y, x


def evaluate_performance(y_test, y_score, y_pred, alpha):
    # evaluation
    n_classes = number_of_clades
    perf = dict()

    # Compute macro-averaged AUPR
    perf["M-aupr"] = 0.0
    n = 0
    for i in range(n_classes):
        perf[i], _, _, _ = real_AUPR(y_test[:, i], y_score[:, i])
        if sum(y_test[:, i]) > 0:
            n += 1
            perf["M-aupr"] += perf[i]
    if n == 0:
        n = 1
    perf["M-aupr"] /= n

    # p, r, f, s = precision_recall_fscore_support(y_test, y_pred, average="macro")
    # print(np.trapz([p], [r]))
    # perf["M-aupr"] = np.trapz(np.asarray([p]), np.asarray([r]))
    # print(perf["M-aupr"])

    # Compute micro-averaged AUPR
    pr, _, _, _ = real_AUPR(y_test, y_score)
    perf["m-aupr"] = pr if pr == pr else 0

    # Computes accuracy
    perf['acc'] = accuracy_score(y_test, y_pred)

    # Computes F1-score
    perf["F1"] = f1_score(y_test, y_pred, average='micro')
    return perf


def train(datafile, row, col, axes, fig):
    # read data file and get embeddings of sequences
    # 'data.csv'
    data = pd.read_csv(datafile, usecols=['sequence', 'clade'])

    # data_val = pd.read_csv('../hollingshed.csv', usecols=['sequence', 'clade'])

    # get number of clades

    global number_of_clades
    for clade in data['clade']:
        if clade > number_of_clades:
            number_of_clades = clade
    data = seq_to_vectors(data)
    # data_val = seq_to_vectors(data_val)

    # split data in train test set
    train_set, test_set = train_test_split(data, shuffle=True, test_size=0.15, train_size=0.85)
    x_train, y_train = train_set['seq_emb'], np.array(train_set['clade'].astype(int))
    x_test, y_test = test_set['seq_emb'], np.array(test_set['clade'].astype(int))
    # x_val, y_val = data_val['seq_emb'], np.array(data_val['clade'].astype(int))
    number_test_clades = max(y_test)
    # number_val_clades = max(y_val)

    train_len = len(x_train)

    # convert labels to proper format for training if y =4 than [0,0,0,1,0,0]
    y_temp = np.zeros((len(y_train), number_of_clades), dtype=int)
    for index in range(len(y_train)):
        temp = y_temp[index]
        temp[y_train[index]-1] = 1
        y_temp[index] = temp
    y_train = y_temp

    y_temp = np.zeros((len(y_test), number_of_clades), dtype=int)
    for index in range(len(y_test)):
        temp = y_temp[index]
        temp[y_test[index]-1] = 1
        y_temp[index] = temp
    y_test = y_temp

    # y_temp = np.zeros((len(y_val), number_of_clades), dtype=int)
    # for index in range(len(y_val)):
    #     temp = y_temp[index]
    #     temp[y_val[index]-1] = 1
    #     y_temp[index] = temp
    # y_val = y_temp

    classes = []
    for i in range(1,number_of_clades+1):
        classes.append(i)

    # select the configured binding modes
    selected_classes = np.arange(number_of_clades)
    classes = list(map(classes.__getitem__, selected_classes))
    y_train = y_train[:, selected_classes]
    num_classes = len(classes)
    y_train_temp = y_train

    x_train = create_matrices(x_train)
    x_train_temp = x_train
    x_test = create_matrices(x_test)

    # x_val = create_matrices(x_val)

    model = None
    best_model = model
    best_vectors = None
    split_len = len(x_train)

    best_alpha = 0
    max_micro_aupr = 0
    max_acc = 0
    c_penalty = 10


    macro_aupr_mean = []
    micro_aupr_mean = []
    acc_mean = []
    f1_mean = []

    # start training here
    for alpha in tqdm(range(0, 10, 1)):
        f_macro_aupr = []
        f_micro_aupr = []
        f_acc = []
        f_f1 = []

        X_train, X_test, _ = build_kernal_matrix(x_train, x_test, 100, alpha/10)
        y_labels = y_train
        for seed in range(10):
            macro_aupr = []
            micro_aupr = []
            acc = []
            f1 = []
            counter = 1
            splits = ml_split(y_labels, num_splits=10, seed=seed)

            for train, valid in splits:
                x_train_now = np.nan_to_num(X_train[train, :][:, train])
                x_valid = np.nan_to_num(X_train[valid, :][:, train])

                y_train_now = y_labels[train.astype(int)]
                y_valid = y_labels[valid.astype(int)]

                model = RandomForestClassifier()
                # model = OneVsRestClassifier(svm.SVC(C=c_penalty, kernel='precomputed', random_state=42, probability=True), n_jobs=-1)
                model.fit(x_train_now, y_train_now)

                # y_score_valid = np.array(model.predict_proba(x_valid))
                y_score_valid = np.array(model.predict_proba(x_valid))[:,:,1].T
                y_pred_valid = model.predict(x_valid)
                result = evaluate_performance(y_valid, y_score_valid, y_pred_valid, alpha)

                micro_aupr.append(result['m-aupr'])
                macro_aupr.append(result['M-aupr'])
                f1.append(result['F1'])
                acc.append(result['acc'])
                counter += 1

            f_micro_aupr.append(round(np.mean(micro_aupr), 3))
            f_macro_aupr.append(round(np.mean(macro_aupr), 3))
            f_f1.append(round(np.mean(f1), 3))
            f_acc.append(round(np.mean(acc), 3))

        # choose alpha based on macro-AUPR
        micro_aupr_mean.append(round(np.mean(f_micro_aupr), 3))
        macro_aupr_mean.append(round(np.mean(f_macro_aupr), 3))
        acc_mean.append(round(np.mean(f_acc), 3))
        f1_mean.append(round(np.std(f_f1), 3))
        if max_micro_aupr <= round(np.mean(f_micro_aupr), 3) and max_acc <= round(np.mean(f_acc), 3):
            max_micro_aupr = round(np.mean(f_micro_aupr), 3)
            max_acc = round(np.mean(f_acc), 3)
            best_alpha = alpha / 10

    x_train = x_train_temp
    y_train = y_train_temp
    x_train, x_test, vectors = build_kernal_matrix(x_train, x_test, 100, best_alpha, True)
    best_vectors = vectors

    # _, x_val, _ = build_kernal_matrix(x_train_temp, x_val, 100, best_alpha, True)

    model = RandomForestClassifier()
    # model = OneVsRestClassifier(svm.SVC(C=c_penalty, kernel='precomputed', random_state=42, probability=True), n_jobs=-1)
    model.fit(x_train, y_train)

    # model = best_model

    y_pred = model.predict(x_test)
    y_score = np.array(model.predict_proba(x_test))[:,:,1].T
    # y_score = np.array(model.predict_proba(x_test))
    print('f1-score is: ', f1_score(y_test, y_pred, average='micro'))
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

    title = ""
    title2 = ""
    if len_data == 2 or len_data==1:
        if datafile[:-4] == "europe_CDR":
            title = "Europe PspA"
            title2 = "Europe_PspA"
        elif datafile[:-4] == "entire_CDR":
            title = "Published PspA"
            title2 = "Published_PspA"
        elif datafile[:-4] == "../published":
            title = "Published CDR PspA"
            title2 = "Published_CDR_PspA"
        elif datafile[:-4] == "../full_length_published":
            title = "Published Full Length PspA"
            title2 = "Published_full_length_PspA"
    else:
        if datafile[:-4] == "europe_CDR":
            title = "CDR"
            title2 = "CDR"
        elif datafile[:-4] == "alpha_helical":
            title = "Alpha Helical"
            title2 = "Alpha_Helical"
        elif datafile[:-4] == "choline_binding":
            title = "Choline Binding"
            title2 = "Choline_Binding"
        elif datafile[:-4] == "proline_rich":
            title = "Proline Rich"
            title2 = "Proline_Rich"
        elif datafile[:-4] == "full_length_europe":
            title = "Full length"
            title2 = "Full_length"

    pickle.dump(model, open("rf_model_" + title2 + ".pkl", "wb"))
    best_vectors['train_length'] = split_len
    best_vectors['alpha'] = best_alpha
    pickle.dump(best_vectors, open("../vectors/rf_vectors_" + title2 + ".txt", "wb"))

    fig2, cfp = plt.subplots()
    sns.heatmap(cm, annot=True, cmap="Purples", ax=cfp, fmt='d')
    cfp.set_title(title + ' sequences Confusion Matrix')
    cfp.set_xlabel('Predicted Clades')
    cfp.set_ylabel('Actual Clades')
    cfp.xaxis.set_ticklabels(range(1, number_test_clades + 1))
    cfp.yaxis.set_ticklabels(range(1, number_test_clades + 1))
    fig2.show()
    fig2.savefig('../plots/RF_Covariance_' + title2 + '.png')

    # hollingshed validation

    # y_pred_val = model.predict(x_val)
    # y_score_val = np.array(model.predict_proba(x_val))[:,:,1].T
    # # y_score = np.array(model.predict_proba(x_test))
    # print('f1-score is: ', f1_score(y_val, y_pred_val, average='micro'))
    # cmh = confusion_matrix(y_val.argmax(axis=1), y_pred_val.argmax(axis=1))

    # fig2, cfp = plt.subplots()
    # sns.heatmap(cmh, annot=True, cmap="Purples", ax=cfp, fmt='d')
    # cfp.set_title('Hollingshed sequences Confusion Matrix')
    # cfp.set_xlabel('Predicted Clades')
    # cfp.set_ylabel('Actual Clades')
    # cfp.xaxis.set_ticklabels(range(1, number_val_clades + 1))
    # cfp.yaxis.set_ticklabels(range(1, number_val_clades + 1))
    # fig2.show()
    # fig2.savefig('../plots/RF_Covariance_Hollingshed.png')

    # precision_val = []
    # recall_val = []
    # fig1, axes1 = plt.subplots()
    # fig1.suptitle("RF AUPR Curve Hollingshed Sequences")
    # for i in range(number_of_clades):
    #     pr, _, precision_val, recall_val = real_AUPR(y_val[:, i], y_score_val[:, i])
    #     axes1.plot(recall_val, precision_val, lw=1, label="class {}".format(i + 1)+", {:.1f}".format(round(pr, 3)*100)+" %")
    #     axes1.set(xlabel='recall', ylabel='precision')
    #     axes1.legend(loc="best", prop={'size': 7})
    # # fig1.tight_layout()
    # # fig1.delaxes(axes1[1])
    # plt.show()
    # fig1.savefig('../plots/RF_Hollingshed_AUPR.png')

    precision = []
    recall = []
    for i in range(number_of_clades):
        pr, _, precision, recall = real_AUPR(y_test[:, i], y_score[:, i])
        if len_data == 1:
            axes.plot(recall, precision, lw=1, label="class {}".format(i + 1)+", {:.1f}".format(round(pr, 3)*100)+" %")
        elif len_data == 2:
            axes[col].plot(recall, precision, lw=1, label="class {}".format(i + 1)+", {:.1f}".format(round(pr, 3)*100)+" %")
        else:
            axes[row, col].plot(recall, precision, lw=1, label="class {}".format(i + 1)+", {:.1f}".format(round(pr, 3)*100)+" %")


    if len_data == 2:
        axes[col].set_title(title)
    elif len_data > 2:
        axes[row, col].set_title(title)


def main():
    # embedding_file = ['protvec/vector_embeddings_europe.txt', 'protvec/vector_embeddings_alpha_helical.txt', 'protvec/vector_embeddings_choline_binding.txt', 'protvec/vector_embeddings_proline_rich.txt', 'protvec/vector_embeddings_whole.txt']#, 'protvec/vector_embeddings_all.txt']
    # data_file = ['europe_CDR.csv', 'alpha_helical.csv', 'choline_binding.csv', 'proline_rich.csv', 'full_length_europe.csv']#, 'data_CDR.csv']
    # embedding_file = ['protvec/vector_embeddings_europe.txt', 'protvec/vector_embeddings_all.txt']
    # data_file = ['europe_CDR.csv', 'entire_CDR.csv']
    embedding_file = ['../protvec/vector_embeddings_full_length_published.txt']
    data_file = ['../full_length_published.csv']
    global len_data
    len_data = len(data_file)
    if len_data == 1:
        fig, axes = plt.subplots()
        fig.suptitle("RF AUPR Curve Published Full Length Region")
    elif len_data == 2:
        fig, axes = plt.subplots(1, 2)
        fig.suptitle("RF AUPR Curve Published CDR Region")
    else:
        fig, axes = plt.subplots(3, 2)
        fig.suptitle("RF AUPR Curve Europe Sequences")
    row = 0
    col = 0
    for i in range(len(data_file)):
        read_embeddings(embedding_file[i])
        train(data_file[i], row, col, axes, fig)
        col += 1
        if col % 2 == 0:
            col = 0
            row += 1
    if len_data == 1:
        axes.set(xlabel='recall', ylabel='precision')
        axes.legend(loc="best", prop={'size': 7})
    else:
        for ax in axes.flat:
            ax.set(xlabel='recall', ylabel='precision')
            if len_data == 2:
                ax.legend(loc="best", prop={'size': 7})
            else:
                ax.legend(loc="best", prop={'size': 4})
    fig.tight_layout()
    # fig.delaxes(axes[0][1])
    plt.show()
    fig.savefig('../plots/RF_Published_Full_Length.png')



if __name__ == '__main__':
    main()

