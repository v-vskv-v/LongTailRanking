import numpy as np
import re
import gzip
import shutil
from tqdm import tqdm

import tensorflow_hub as hub
import fasttext
import pymorphy2

from sklearn.metrics.pairwise import cosine_similarity as cosine

import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS


nltk.download('stopwords')
stop_words = set(stopwords.words('russian') + stopwords.words('english')) | ENGLISH_STOP_WORDS

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
morph = pymorphy2.MorphAnalyzer()

with gzip.open('data/fast_text/cc.ru.300.bin.gz', 'rb') as f_in,\
        open('cc.ru.300.bin', 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)

ft_ru = fasttext.load_model('cc.ru.300.bin')

dict_queries = dict()
dict_titles = dict()
dict_queries_b = dict()
dict_titles_b = dict()
words_emb_dict = {}
X_split = []
query_split = []


def get_words_sim_dict(string_q, string_t, emb_dict):
    res_dict = dict()
    for word_q in string_q.split(' '):
        a = emb_dict[word_q]
        for word_t in string_t.split(' '):
            b = emb_dict[word_t]
            res_dict['{}\t{}'.format(word_q, word_t)] = cosine(a, b)[0][0]
    return res_dict


def get_result(query, title, res_dict):
    sorted_dict = sorted(res_dict.items(), key=lambda kv: -1*kv[1])
    result = 0.0
    nums = 0.0
    for x in sorted_dict:
        words = x[0].split('\t')
        score = x[1]
        if words[0] in query and words[1] in title:
            result += score
            nums += 1
            query = query.replace(words[0], '')
            title = title.replace(words[1], '')
        if nums == 0.0:
            return 0.0
    return result / nums


def get_features(path, dict_emb):
    with open(path, 'w', encoding='utf-8') as f:
        for i in tqdm(range(len(X_split))):
            res_ = 0.0
            if (X_split[i] in dict_titles) and (query_split[i] in dict_queries):
                query = dict_queries[query_split[i]]
                title = dict_titles[X_split[i]]
                res_dict = get_words_sim_dict(query, title, dict_emb)
                res_ = get_result(query, title, res_dict)
            f.write('{}\t{}\t{]\n'.format(query_split[i], X_split[i], str(res_)))


def make_pair(string):
    tmp = string.strip().split()
    tmp_p = []
    for i in range(0, len(tmp)-1, 2):
        tmp_p.append('{} {}'.format(tmp[i], tmp[i+1]))
    if len(tmp) % 2 == 1:
        tmp_p.append(tmp[-1])
    return tmp_p


with open('data/queries_b_proc.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        if tmp[0] not in dict_queries:
            string_arr = [i for i in tmp[1] if i not in stop_words]
            norm_form = ' '.join(string_arr)
            dict_queries[tmp[0]] = norm_form

with open('data/queries_b.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        if tmp[0] not in dict_queries_b:
            dict_queries_b[tmp[0]] = tmp[1]

with open('data/titles_norm.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        string_arr = [i for i in tmp[1].lower().split(' ') if i not in stop_words]
        dict_titles[tmp[0]] = ' '.join(string_arr)

with open('data/titles.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        if len(tmp) == 2:
            dict_titles_b[tmp[0]] = tmp[1].lower()

with open('data/train.marks.tsv', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        query_split.append(tmp[0])
        X_split.append(tmp[1])

with open('data/sample.csv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f.readlines():
        tmp = line.strip().split(',')
        query_split.append(tmp[0])
        X_split.append(tmp[1])

X_split = np.array(X_split)
query_split = np.array(query_split)

with open('features/USE_b.txt', 'w') as f_nonnorm,\
        open('features/USE.txt', 'w') as f_:
    for qid in np.unique(query_split):
        if qid in dict_queries_b:
            tid = X_split[np.argwhere(query_split == qid).ravel()]
            titles = [' ' if id_ not in dict_titles_b else dict_titles_b[id_] for id_ in tid]
            res = cosine(embed([dict_queries_b[qid]]), embed(titles)).ravel()
            for i in range(len(tid)):
                f_nonnorm.write('{}\t{}\t{}\n'.format(qid, tid[i], str(res[i])))
        if qid in dict_queries:
            tid = X_split[np.argwhere(query_split == qid).ravel()]
            titles = [' ' if id_ not in dict_titles else dict_titles[id_] for id_ in tid]
            res = cosine(embed([dict_queries[qid]]), embed(titles)).ravel()
            for i in range(len(tid)):
                f_.write('{}\t{}\t{}\n'.format(qid, tid[i], str(res[i])))

title_query = ' '.join([dict_titles[i] for i in dict_titles]) + ' ' + ' '.join([dict_queries[i] for i in dict_queries])
words_emb_dict = dict()

for word in np.unique(title_query.split()):
    if word not in words_emb_dict:
        words_emb_dict[word] = embed(word)

get_features('features/USE_byword.txt', embed)
get_features('features/ff.txt', ft_ru)
