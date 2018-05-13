import fasttext
import numpy as np


class FasttextLoader:
    def __init__(self):
        self.vocabulary = {}
        self.w2v = None
        self.count = 1
        self.weights_were_counted = False

    def load(self, filename):
        if not filename.endswith(".bin"):
            raise ValueError("w2v must be with extension .bin")

        self.w2v = fasttext.load_model(filename)

        for word in self.w2v.words:
            self.vocabulary[word] = (self.w2v[word], self.count)
            self.count += 1

    def get_index(self, word):
        if self.weights_were_counted:
            raise ValueError("Weights have been already counted. Create a new w2v and EmbeddingLayer")

        if word not in self.vocabulary:
            self.vocabulary[word] = (self.w2v[word], self.count)
            self.count += 1
        return self.vocabulary[word][1]

    def vocabulary_size(self):
        return len(self.vocabulary)

    def vector_dim(self):
        return self.w2v.dim

    def weights(self):
        zero_vector = np.zeros(self.vector_dim())
        s = sorted(self.vocabulary.items(), key=lambda x: x[1][1])
        s = np.asarray([zero_vector] + [elem[1][0] for elem in s])
        self.weights_were_counted = True
        return s
