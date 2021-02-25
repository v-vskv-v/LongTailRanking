from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as cosine
import pymorphy2
from rank_bm25 import BM25Okapi
from scipy import spatial
import tqdm
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS


nltk.download('stopwords')
stop_words = set(stopwords.words('russian') + stopwords.words('english')) | ENGLISH_STOP_WORDS
morph = pymorphy2.MorphAnalyzer()

N = 6311

num2q_dict = dict()
num2title_dict = dict()
q2url_dict = dict()


def tf_idf_docs(q):
    corpus = []
    query = []
    dict_title = dict()

    with open('data/norm_docs2q/{}.txt'.format(q), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tmp = line.strip().split('\t')
            corpus.append(tmp[1])
            dict_title[len(corpus)-1] = tmp[0]

    with open('data/queries_b_proc.txt', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            tmp = line.strip().split('\t')
            if tmp[0] == q:
                query.append(tmp[1])

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,7))
    a = corpus + query
    vectorizer.fit(a)
    vec_docs = vectorizer.transform(corpus)
    vec_qu = vectorizer.transform(query)
    res = cosine(vec_qu, vec_docs).ravel()

    with open('features/tfidf_body/{}.txt'.format(q), 'w', encoding='utf-8') as f:
        for i in range(len(res)):
            f.write('{}\t{}\t{}\n'.format(q, dict_title[i], str(res[i])))


def tf_idf_titles(title, a, b):
    with open('features/tfidf_char{}_{}.txt'.format(a, b), 'w') as f:
        if a == 1 and b == 1:
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(a, b), encoding='utf-8')
            vectorizer.fit(title)
            for q in tqdm(q2url_dict):
                if q not in num2q_dict:
                    continue
                b = vectorizer.transform([num2q_dict[q]]).toarray().ravel()
                for d in q2url_dict[q]:
                    if d not in num2title_dict:
                        continue
                    a = vectorizer.transform([num2title_dict[d]]).toarray().ravel()
                    cos_sim = 1-spatial.distance.cosine(a, b)
                    f.write('{}\t{}\t{}\n'.format(q, d, str(cos_sim)))
        else:
            for q in tqdm(q2url_dict):
                if q not in num2q_dict:
                    continue
                docs = []
                for d in q2url_dict[q]:
                    if d not in num2title_dict:
                        continue
                    docs.append(num2title_dict[d])
                vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(a,b), encoding='utf-8')
                vectorizer.fit(docs)
                b = vectorizer.transform([num2q_dict[q]]).toarray().ravel()
                for d in q2url_dict[q]:
                    if d not in num2title_dict:
                        continue
                    a = vectorizer.transform([num2title_dict[d]]).toarray().ravel()
                    cos_sim = 1 - spatial.distance.cosine(a, b)
                    f.write('{}\t{}\t{}\n'.format(q, d, str(cos_sim)))


def remove_stopwords(string):
    word_list = string.split(' ')
    filtered_words = [word for word in word_list if word not in stop_words]
    filtered_string = ' '.join(filtered_words)
    return filtered_string


def convert_to_ngramms(string):
    chars = list(string.replace(' ', ''))
    bigrams = [chars[i]+chars[i+1] for i in range(len(chars)-1)]
    trigrams = [chars[i]+chars[i+1]+chars[i+2] for i in range(len(chars)-2)]
    return chars+bigrams+trigrams


with open('data/queries_b_proc.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        if tmp[0] not in num2q_dict:
            num2q_dict[tmp[0]] = tmp[1]

with open('data/titles_norm.txt', 'r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        num2title_dict[tmp[0]] = tmp[1]

with open('data/train.marks.tsv','r') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        if tmp[0] in q2url_dict:
            q2url_dict[tmp[0]].append(tmp[1])
        else:
            q2url_dict[tmp[0]] = [tmp[1]]

with open('data/sample.csv','r') as f:
    f.readline()
    for line in f.readlines():
        tmp = line.strip().split(',')
        if tmp[0] in q2url_dict:
            q2url_dict[tmp[0]].append(tmp[1])
        else:
            q2url_dict[tmp[0]] = [tmp[1]]

title = []
for t in num2title_dict:
    title.append(num2title_dict[t])

tf_idf_titles(title, 1, 1)
tf_idf_titles(title, 2, 10)

for i in range(N):
    tf_idf_docs(str(i))

# === === === #

with open('features/сosine_bm25.txt', 'w') as f:
    for q in tqdm(q2url_dict):
        if q not in num2q_dict:
            continue
        docs = []
        for d in q2url_dict[q]:
            if d not in num2title_dict:
                continue
            docs.append(remove_stopwords(num2title_dict[d]))
        q_new = remove_stopwords(num2q_dict[q])
        tokenized_corpus = [convert_to_ngramms(doc) for doc in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = convert_to_ngramms(q_new)
        doc_scores = bm25.get_scores(tokenized_query)
        for d in q2url_dict[q]:
            if d not in num2title_dict:
                continue
            f.write('{}\t{}\t{}\n'.format(q, d, str(doc_scores[i])))

