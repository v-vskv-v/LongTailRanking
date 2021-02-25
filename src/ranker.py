import numpy as np
from scipy.sparse import csr_matrix
from preprocess import take_part

import warnings
import copy
warnings.filterwarnings('ignore')

import lightgbm as lgbm

PAD = 39
id_ = 0
dict_train = dict()
X_dict = dict()
host_dict = dict()
fasttext_dict = dict()
url2host_dict = dict()
tf = dict()
dict_USE_words = dict()
dict_url_qu = dict()
dict_host_qu = dict()
dict_host = dict()
dict_host_qu = dict()
dict_qu = dict()
dict_sDBN_qu = dict()
dict_sDBN_qh = dict()
dict_sDBN_h = dict()
dict_sDBN_u = dict()

y_train = []
query_train = []
X_train_split = []
X_train = []
query_test = []
y_test = []
X_test_split = []
X_test = []

with open('data/train.marks.tsv', 'r') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t');
        dict_train[tmp[1]] = 1

with open('data/train.marks.tsv', 'r') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        y_train.append(np.int(tmp[2]))
        query_train.append(np.int(tmp[0]))
        X_train_split.append(tmp[1])


for d in X_train_split:
    if d in X_dict:
        X_train.append(X_dict[d])
    else:
        X_train.append(list(np.zeros(39)))

query_train = np.array(query_train)
y_train = np.array(y_train)
X_train = np.array(X_train)

with open('data/sample.csv', 'r') as f:
    f.readline()
    for line in f.readlines():
        tmp = line.strip().split(',')
        query_test.append(int(tmp[0]))
        y_test.append(1)
        X_test_split.append(tmp[1])

for d in X_test_split:
    if d in X_dict:
        X_test.append(X_dict[d])
    else:
        X_test.append(list(np.zeros(PAD)))

query_test = np.array(query_test)
y_test = np.array(y_test)
X_test = np.array(X_test)

with open('data/url.data', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        tmp_ = tmp[1].split('/')[0]
        if tmp_[:4] == 'www.':
            tmp_=tmp_[4:]
        if tmp_ in host_dict:
            url2host_dict[tmp[0]] = host_dict[tmp_]
            continue
        host_dict[tmp_] = str(id_)
        url2host_dict[tmp[0]] = host_dict[tmp_]
        id_ += 1

arr_query_unic = []
with open('data/sample.csv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f.readlines():
        tmp = line.strip().split(',')
        arr_query_unic.append(tmp[0])
with open('data/train.marks.tsv', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        arr_query_unic.append(tmp[0])
arr_query_unic = np.unique(arr_query_unic)

for q in arr_query_unic:
    with open('features/tf_idf_body/' + q + '.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tmp = line.strip().split('\t')
            tf[tmp[0]+'\t'+tmp[1]] = tmp[2]

a_tf = np.zeros((len(X_train_split), 1))
for i in range(len(X_train_split)):
    if (str(query_train[i])+'\t'+X_train_split[i]) in tf:
        a_tf[i] = tf[str(query_train[i])+'\t'+X_train_split[i]]

b_tf = np.zeros((len(X_test_split), 1))
for i in range(len(X_test_split)):
    if (str(query_test[i])+'\t'+X_test_split[i]) in tf:
        b_tf[i] = tf[str(query_test[i])+'\t'+X_test_split[i]]

tf_title_2_10 = dict()
with open('features/tfidf_char2_10.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        tf_title_2_10[tmp[0]+'\t'+tmp[1]] = tmp[2]

a_tf_title_2_10 = np.zeros((len(X_train_split), 1))
for i in range(len(X_train_split)):
    if (str(query_train[i])+'\t'+X_train_split[i]) in tf_title_2_10:
        a_tf_title_2_10[i] = tf_title_2_10[str(query_train[i])+'\t'+X_train_split[i]]

b_tf_title_2_10 = np.zeros((len(X_test_split), 1))
for i in range(len(X_test_split)):
    if (str(query_test[i])+'\t'+X_test_split[i]) in tf_title_2_10:
        b_tf_title_2_10[i] = tf_title_2_10[str(query_test[i])+'\t'+X_test_split[i]]

tf_title_1_1 = dict()
with open('features/tfidf_char1_1.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        tf_title_1_1[tmp[0]+'\t'+tmp[1]] = tmp[2]

a_tf_title_1_1 = np.zeros((len(X_train_split), 1))
for i in range(len(X_train_split)):
    if (str(query_train[i])+'\t'+X_train_split[i]) in tf_title_1_1:
        a_tf_title_1_1[i] = tf_title_1_1[str(query_train[i])+'\t'+X_train_split[i]]

b_tf_title_1_1 = np.zeros((len(X_test_split), 1))
for i in range(len(X_test_split)):
    if (str(query_test[i])+'\t'+X_test_split[i]) in tf_title_1_1:
        b_tf_title_1_1[i] = tf_title_1_1[str(query_test[i])+'\t'+X_test_split[i]]

with open('features/ff.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        fasttext_dict[tmp[0]+'\t'+tmp[1]] = np.float64(tmp[2])

a_fasttext = np.zeros((len(X_train_split), 1))
for i in range(len(X_train_split)):
    if (str(query_train[i])+'\t'+X_train_split[i]) in fasttext_dict:
        a_fasttext[i] = fasttext_dict[str(query_train[i])+'\t'+X_train_split[i]]

b_fasttext = np.zeros((len(X_test_split), 1))
for i in range(len(X_test_split)):
    if (str(query_test[i])+'\t'+X_test_split[i]) in fasttext_dict:
        b_fasttext[i] = fasttext_dict[str(query_test[i])+'\t'+X_test_split[i]]

bm25_dict = dict()
with open('features/сosine_bm25.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
      tmp = line.strip().split('\t')
      bm25_dict[tmp[0]+'\t'+tmp[1]] = tmp[2]

a_bm25_dict = np.zeros((len(X_train_split), 1))
for i in range(len(X_train_split)):
    if (str(query_train[i])+'\t'+X_train_split[i]) in bm25_dict:
        a_bm25_dict[i] = bm25_dict[str(query_train[i])+'\t'+X_train_split[i]]

b_bm25_dict = np.zeros((len(X_test_split), 1))
for i in range(len(X_test_split)):
    if (str(query_test[i])+'\t'+X_test_split[i]) in bm25_dict:
        b_bm25_dict[i] = bm25_dict[str(query_test[i])+'\t'+X_test_split[i]]

with open('features/USE_byword.txt', 'r') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        dict_USE_words[tmp[0]+'\t'+tmp[1]] = tmp[2]

a_USE_words = np.zeros((len(X_train_split), 1))
for i in range(len(X_train_split)):
    if (str(query_train[i])+'\t'+X_train_split[i]) in dict_USE_words:
        a_USE_words[i] = dict_USE_words[str(query_train[i])+'\t'+X_train_split[i]]

b_USE_words = np.zeros((len(X_test_split), 1))
for i in range(len(X_test_split)):
    if (str(query_test[i])+'\t'+X_test_split[i]) in dict_USE_words:
        b_USE_words[i] = dict_USE_words[str(query_test[i])+'\t'+X_test_split[i]]

dict_USE_norm = dict()
with open('features/USE.txt', 'r') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        dict_USE_norm[tmp[0]+'\t'+tmp[1]] = tmp[2]

a_USE_norm = np.zeros((len(X_train_split), 1))
for i in range(len(X_train_split)):
    if (str(query_train[i])+'\t'+X_train_split[i]) in dict_USE_norm:
        a_USE_norm[i] = dict_USE_norm[str(query_train[i])+'\t'+X_train_split[i]]

b_USE_norm = np.zeros((len(X_test_split), 1))
for i in range(len(X_test_split)):
    if (str(query_test[i])+'\t'+X_test_split[i]) in dict_USE_norm:
        b_USE_norm[i] = dict_USE_norm[str(query_test[i])+'\t'+X_test_split[i]]

dict_USE_b = dict()
with open('features/USE_b.txt', 'r') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        dict_USE_b[tmp[0]+'\t'+tmp[1]] = tmp[2]

a_USE_b = np.zeros((len(X_train_split), 1))
for i in range(len(X_train_split)):
  if (str(query_train[i])+'\t'+X_train_split[i]) in dict_USE_b:
    a_USE_b[i] = dict_USE_b[str(query_train[i])+'\t'+X_train_split[i]]

b_USE_b = np.zeros((len(X_test_split), 1))
for i in range(len(X_test_split)):
  if (str(query_test[i])+'\t'+X_test_split[i]) in dict_USE_b:
    b_USE_b[i] = dict_USE_b[str(query_test[i])+'\t'+X_test_split[i]]

with open('features/results_u/part-r-00000', 'r') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        X_dict[tmp[0]] = [np.float64(i) for i in tmp[1:]]

for i in range(11):
    with open('features/results_uq/part-r-000'+take_part(i), 'r') as f:
        for line in f.readlines():
            tmp = line.strip().split('\t')
            dict_url_qu[tmp[0]+'\t'+tmp[1]] = [np.float64(el) for el in tmp[2:]]

a_url_qu = np.zeros((len(X_train_split), PAD))
for i in range(len(X_train_split)):
    if (str(query_train[i])+'\t'+X_train_split[i]) in dict_url_qu:
        a_url_qu[i] = dict_url_qu[str(query_train[i])+'\t'+X_train_split[i]]

b_url_qu = np.zeros((len(X_test_split), PAD))
for i in range(len(X_test_split)):
    if (str(query_test[i])+'\t'+X_test_split[i]) in dict_url_qu:
        b_url_qu[i] = dict_url_qu[str(query_test[i])+'\t'+X_test_split[i]]

for i in range(11):
    with open('features/results_hq/part-r-000'+str(i), 'r') as f:
        for line in f.readlines():
            tmp = line.strip().split('\t')
            dict_host_qu[tmp[0]+'\t'+tmp[1]] = [np.float64(el) for el in tmp[2:]]

a_host_qu = np.zeros((len(X_train_split), PAD))
for i in range(len(X_train_split)):
    if (str(query_train[i])+'\t'+url2host_dict[X_train_split[i]]) in dict_host_qu:
        a_host_qu[i] = dict_host_qu[str(query_train[i])+'\t'+url2host_dict[X_train_split[i]]]

b_host_qu = np.zeros((len(X_test_split), PAD))
for i in range(len(X_test_split)):
    if (str(query_test[i])+'\t'+url2host_dict[X_test_split[i]]) in dict_host_qu:
        b_host_qu[i] = dict_host_qu[str(query_test[i])+'\t'+url2host_dict[X_test_split[i]]]

with open('features/results_h/part-r-00000', 'r') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        dict_host[tmp[0]] = [np.float64(el) for el in tmp[1:]]

a_host = np.zeros((len(X_train_split), PAD))
for i in range(len(X_train_split)):
    if url2host_dict[X_train_split[i]] in dict_host:
        a_host[i] = dict_host[url2host_dict[X_train_split[i]]]

b_host = np.zeros((len(X_test_split), PAD))
for i in range(len(X_test_split)):
    if url2host_dict[X_test_split[i]] in dict_host:
        b_host[i] = dict_host[url2host_dict[X_test_split[i]]]

with open('features/results_q/part-r-00000', 'r') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        dict_qu[tmp[0]] = [np.float64(el) for el in tmp[1:]]

a_qu = np.zeros((len(X_train_split), 8))
for i in range(len(X_train_split)):
    if str(query_train[i]) in dict_qu:
        a_qu[i] = dict_qu[str(query_train[i])]

b_qu = np.zeros((len(X_test_split), 8))
for i in range(len(X_test_split)):
    if str(query_test[i]) in dict_qu:
        b_qu[i] = dict_qu[str(query_test[i])]

with open('features/results_sDBN/qu/part-r-00000', 'r') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        dict_sDBN_qu[tmp[0]+'\t'+tmp[1]] = [np.float64(el) for el in tmp[2:]]

a_sDBN_qu = np.zeros((len(X_train_split), 3))
for i in range(len(X_train_split)):
    if (str(query_train[i])+'\t'+X_train_split[i]) in dict_sDBN_qu:
        a_sDBN_qu[i] = dict_sDBN_qu[str(query_train[i])+'\t'+X_train_split[i]]

b_sDBN_qu = np.zeros((len(X_test_split), 3))
for i in range(len(X_test_split)):
    if (str(query_test[i])+'\t'+X_test_split[i]) in dict_sDBN_qu:
        b_sDBN_qu[i] = dict_sDBN_qu[str(query_test[i])+'\t'+X_test_split[i]]

with open('features/results_sDBN/qh/part-r-00000', 'r') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        dict_sDBN_qh[tmp[0]+'\t'+tmp[1]] = [np.float64(el) for el in tmp[2:]]

a_sDBN_qh = np.zeros((len(X_train_split), 3))
for i in range(len(X_train_split)):
    if (str(query_train[i])+'\t'+url2host_dict[X_train_split[i]]) in dict_sDBN_qh:
        a_sDBN_qh[i] = dict_sDBN_qh[str(query_train[i])+'\t'+url2host_dict[X_train_split[i]]]

b_sDBN_qh = np.zeros((len(X_test_split), 3))
for i in range(len(X_test_split)):
    if (str(query_test[i])+'\t'+url2host_dict[X_test_split[i]]) in dict_sDBN_qh:
        b_sDBN_qh[i] = dict_sDBN_qh[str(query_test[i])+'\t'+url2host_dict[X_test_split[i]]]

with open('features/results_sDBN/h/part-r-00000', 'r') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        dict_sDBN_h[tmp[0]] = [np.float64(el) for el in tmp[1:]]

a_sDBN_h = np.zeros((len(X_train_split), 3))
for i in range(len(X_train_split)):
    if url2host_dict[X_train_split[i]] in dict_sDBN_h:
        a_sDBN_h[i] = dict_sDBN_h[url2host_dict[X_train_split[i]]]

b_sDBN_h = np.zeros((len(X_test_split), 3))
for i in range(len(X_test_split)):
    if url2host_dict[X_test_split[i]] in dict_sDBN_h:
        b_sDBN_h[i] = dict_sDBN_h[url2host_dict[X_test_split[i]]]

with open('features/results_sDBN/u/part-r-00000', 'r') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        dict_sDBN_u[tmp[0]] = [np.float64(el) for el in tmp[1:]]

a_sDBN_u = np.zeros((len(X_train_split), 3))
for i in range(len(X_train_split)):
    if X_train_split[i] in dict_sDBN_u:
        a_sDBN_u[i] = dict_sDBN_u[X_train_split[i]]

b_sDBN_u = np.zeros((len(X_test_split), 3))
for i in range(len(X_test_split)):
    if X_test_split[i] in dict_sDBN_u:
        b_sDBN_u[i] = dict_sDBN_u[X_test_split[i]]

X_train_ = np.hstack((X_train, a_url_qu, a_host, a_host_qu, a_tf, a_tf_title_1_1, a_tf_title_2_10,  a_fasttext, a_sDBN_u, a_sDBN_qu, a_sDBN_h, a_sDBN_qh, a_qu, a_USE_words, a_USE_norm, a_USE_b, a_bm25_dict))
X_test_ = np.hstack((X_test, b_url_qu, b_host, b_host_qu, b_tf, b_tf_title_1_1, b_tf_title_2_10, b_fasttext, b_sDBN_u, b_sDBN_qu, b_sDBN_h, b_sDBN_qh, b_qu, b_USE_words, b_USE_norm, b_USE_b, b_bm25_dict))

X_train_ = csr_matrix(X_train_)
X_test_ = csr_matrix(X_test_)

group_train = np.unique(query_train, return_counts=True)
train_valid_ids = np.array([i for i in range(len(group_train[0]))])
train_ids = []

train_dataset = []
for qid in group_train[0][train_ids]:
    train_dataset += list(np.argwhere(query_train == qid).ravel())

train_data = lgbm.Dataset(X_train_[train_dataset], label=y_train[train_dataset], group=group_train[1][train_ids])

param = {'objective': 'lambdarank', 'boosting': 'gbdt', 'learning_rate': 0.01}

ranker = lgbm.train(param, train_data, num_boost_round=4500)
result = ranker.predict(X_test_)
X_test_split = np.array(X_test_split)

with open('submission.csv', 'w') as f:
    f.write('QueryId,DocumentId\n')
    for qid in np.unique(query_test):
        q_doc_idxs = np.argwhere(query_test == qid).ravel()
        doc_ids = copy.deepcopy(X_test_split[q_doc_idxs])
        q_doc_scores = result[q_doc_idxs]
        sorted_doc_ids = doc_ids[np.argsort(q_doc_scores)[::-1]]
    for d in sorted_doc_ids:
        f.write('{0},{1}\n'.format(qid, d))
