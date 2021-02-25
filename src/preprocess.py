import pymorphy2
from pyaspeller import YandexSpeller


dict_words2normword = dict()
dict_titles = dict()
dict_titles_norm = dict()
dict_q2d = dict()
host_dict = dict()

morph = pymorphy2.MorphAnalyzer()
speller = YandexSpeller(find_repeat_words=True, ignore_capitalization=True)


def normalization(string):
    words = string.split(' ')
    normwords = []
    for word in words:
        if word in dict_words2normword:
            normwords.append(dict_words2normword[word])
        else:
            normwords.append(morph.parse(word)[0].normal_form)
            dict_words2normword[word] = morph.parse(word)[0].normal_form
    return ' '.join(normwords)


def take_part(n):
    if n < 10:
        return '0' + str(n)
    return str(n)


with open('data/queries.tsv', 'r', encoding='utf-8') as f,\
        open('data/queries_b.tsv', 'w', encoding='utf-8') as f_:
    for line in f:
        num, q = line.strip().split('\t')
        correction = speller.spell(q)
        if list(correction):
            for w in correction:
                if w['s']:
                    q = q.replace(w['word'], w['s'][0])
        f_.write('{}\t{}'.format(num, q))

with open('data/queries_b.txt', 'r', encoding='utf-8') as f, \
        open('data/queries_b_proc.txt', 'w', encoding='utf-8') as f_:
    for line in f.readlines():
        num, q = line.strip().split('\t')
        q_w = q.split(' ')
        norm_form = []
        for w in q_w:
            norm_form.append(morph.parse(w)[0].normal_form)
        norm_form = ' '.join(norm_form)
        f_.write('{}\t{}'.format(num, norm_form))

with open('data/docs.tsv','r', encoding='utf-8') as f:
    for line in f.readlines():
        tmp = line.strip().split('\t')
        with open('data/norm_docs/{}.txt'.format(tmp[0]), 'w', encoding='utf-8') as f_doc:
            if len(tmp) == 3:
                dict_titles[tmp[0]] = tmp[1].lower()
                dict_titles_norm[tmp[0]] = normalization(tmp[1].lower())
                f_doc.write('{}\t{}'.format(tmp[0], normalization(tmp[2].lower())))
            else:
                dict_titles[tmp[0]] = ''
                dict_titles_norm[tmp[0]] = ''
                f_doc.write('{}\t{}'.format(tmp[0], normalization(tmp[1].lower())))

with open('data/titles.txt', 'w', encoding='utf-8') as f_title:
    for d in dict_titles:
        f_title.write('{}\t{}'.format(d, dict_titles[d]))

with open('data/titles_norm.txt', 'w', encoding='utf-8') as f_title:
    for d in dict_titles:
        f_title.write('{}\t{}'.format(d, dict_titles_norm[d]))


with open('data/sample.csv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f.readlines():
        tmp = line.strip().split(',')
        if tmp[0] in dict_q2d:
            dict_q2d[tmp[0]].append(tmp[1])
        else:
            dict_q2d[tmp[0]] = [tmp[1]]

with open('data/train.marks.tsv', 'r', encoding='utf-8') as f:
    f.readline()
    for line in f.readlines():
        tmp = line.strip().split('\t')
        if tmp[0] in dict_q2d:
            dict_q2d[tmp[0]].append(tmp[1])
        else:
            dict_q2d[tmp[0]] = [tmp[1]]

for q in dict_q2d:
    with open('data/norm_docs2q/{}.txt'.format(q), 'w', encoding='utf-8') as f:
        for d in dict_q2d[q]:
            try:
                with open('data/norm_docs/{}.txt'.format(d), 'r', encoding='utf-8') as f_:
                    line = f_.readline()
                    f.write(line+'\n')
            except:
                pass
