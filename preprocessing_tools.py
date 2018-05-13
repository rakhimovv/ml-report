import json
import re

import numpy as np
import pymorphy2
from keras.preprocessing import sequence
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from dictfeaturizer import DictFeaturizer


def text_to_words(text):
    return re.findall(r'[абвгдеёжзийклмнопрстуфхцчшщъыьэюя]+', text.lower())


def pp_text(text):
    return ' '.join(text_to_words(text))


def map_indices(text, w2v):
    return [w2v.get_index(word) for word in text.split()]

def normalize_words(words, morph):
    return [morph.parse(t)[0].normal_form for t in words]

def inds_texts_labels(filename, w2v, field="text", min_len=2, multilabel=True):
    inds = []
    texts = []
    labels = []
    data_set = set()
    short_count = 0
    duplicates_count = 0
    not_xeno_count = 0
    multilabel_count = 0

    with open(filename, encoding="utf-8") as f:
        for i, line in enumerate(f):
            elem = json.loads(line)
            text = map_indices(pp_text(elem[field]), w2v)
            if len(text) < min_len:
                short_count += 1
                continue

            label = elem["nation_label"]
            if 0 in label or 1 in label:  # 0, 1 так как not_xeno и unknown соответственно
                not_xeno_count += 1
                continue
            if len(label) > 1:
                multilabel_count += 1
                if not multilabel:
                    continue

            ttext = tuple(text)
            if ttext in data_set:
                duplicates_count = duplicates_count + 1
                continue

            inds.append(text)
            texts.append(elem[field])
            labels.append(label)
            data_set.add(ttext)

    print("Num of data < min_len: {}".format(short_count))
    print("Num of multilabel data: {}".format(multilabel_count))
    print("Num of not_xeno or unknown: {}".format(not_xeno_count))
    print("Num of duplicates: {}".format(duplicates_count))
    return inds, texts, labels


class LemmaTokenizer(object):
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()

    def __call__(self, doc):
        return [self.morph.parse(t)[0].normal_form for t in text_to_words(doc)]


class MultiLabelEncoder(object):
    def __init__(self):
        self.le = LabelEncoder()
        self.mlb = MultiLabelBinarizer()

    def fit(self, y):
        self.le.fit(np.unique([item for sublist in y for item in sublist]))
        self.mlb.fit([self.le.transform(elem) for elem in y])
        return self

    def transform(self, y):
        return self.mlb.transform([self.le.transform(elem) for elem in y])

    def fit_transform(self, y):
        self.le.fit(np.unique([item for sublist in y for item in sublist]))
        return self.mlb.fit_transform([self.le.transform(elem) for elem in y])

    def inverse_transform(self, yt):
        return [self.le.inverse_transform(elem) for elem in self.mlb.inverse_transform(yt)]


class MultilabelStratifiedKFold():
    def __init__(self, targets, n_folds=10):
        self.targets = targets
        self.n_folds = n_folds

    def proba_mass_split(self, y, folds):
        obs, classes = y.shape
        dist = y.sum(axis=0).astype('float')
        dist /= dist.sum()
        index_list = []
        fold_dist = np.zeros((folds, classes), dtype='float')
        for _ in range(folds):
            index_list.append([])
        for i in range(obs):
            if i < folds:
                target_fold = i
            else:
                normed_folds = fold_dist.T / fold_dist.sum(axis=1)
                how_off = normed_folds.T - dist
                target_fold = np.argmin(np.dot((y[i] - .5).reshape(1, -1), how_off.T))
            fold_dist[target_fold] += y[i]
            index_list[target_fold].append(i)
        print("Fold distributions are")
        print(fold_dist)
        return index_list

    def __iter__(self):
        folds = self.proba_mass_split(self.targets, self.n_folds)
        for i in range(self.n_folds):
            train = folds[0:i] + folds[i + 1:]
            test = folds[i]
            yield np.hstack(train), np.array(test)


class W2VTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, weights, dim):
        self.weights = weights
        self.dim = dim

    def transform(self, X, y=None, **fit_params):
        max_length = max([len(elem) for elem in X])
        X = sequence.pad_sequences(X, maxlen=max_length)
        res = np.zeros((len(X), self.dim * max_length))
        for i, inds in enumerate(X):
            for j, ind in enumerate(inds):
                res[i, self.dim * j: self.dim * (j + 1)] = self.weights[ind]
        return res

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self
    

class DictTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, path):
        self.path = path
        self.morph = pymorphy2.MorphAnalyzer()

    def transform(self, X, y=None, **fit_params):
        d = DictFeaturizer.load(self.path)
        res = np.zeros((len(X), 7))
        for i in range(len(X)):
            res[i] = np.array(list(d.transform(normalize_words(text_to_words(X[i]), self.morph)).values()))
        return res

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X)

    def fit(self, X, y=None, **fit_params):
        return self
