import urllib.request
import os
import zipfile
import numpy as np

# http://nlp.stanford.edu/data/glove.6B.zip

def load_glove(data_dir_path=None, embedding_dim=None):
    """
    Parameters
    ----------
    data_dir_path : the directory path where GloVe model is downloaded and stored
    embedding_dim : dimension of embedding (default 300)
    
    Return
    ----------
    word embeddings
    """
    if embedding_dim is None:
        embedding_dim = 300

    glove_file_path = data_dir_path
    
    _word2em = {}
    file = open(glove_file_path, mode='rt', encoding='utf8')
    for line in file:
        words = line.strip().split()
        word = words[0]
        embeds = np.array(words[1:], dtype=np.float32)
        _word2em[word] = embeds
    file.close()
    return _word2em


class GloveModel(object):
   
    model_name = 'glove-model'

    def __init__(self):
        self.word2em = None
        self.embedding_dim = None

    def load(self, data_dir_path, embedding_dim=None):
        if embedding_dim is None:
            embedding_dim = 300
        self.embedding_dim = embedding_dim
        self.word2em = load_glove(data_dir_path, embedding_dim)

    def encode_word(self, word):
        """
        単語のベクトルを返す
        """
        w = word.lower()
        if w in self.word2em:
            return self.word2em[w]
        else:
            return np.zeros(shape=(self.embedding_dim, ))

    def encode_docs(self, docs, max_allowed_doc_length=None):
        '''
        複数の文のベクトルを返す
        '''
        doc_count = len(docs)
        X = np.zeros(shape=(doc_count, self.embedding_dim))
        max_len = 0
        for doc in docs:
            max_len = max(max_len, len(doc.split(' ')))
        if max_allowed_doc_length is not None:
            max_len = min(max_len, max_allowed_doc_length)
        for i in range(0, doc_count):
            doc = docs[i]
            words = [w.lower() for w in doc.split(' ')]
            length = min(max_len, len(words))
            E = np.zeros(shape=(self.embedding_dim, max_len))
            for j in range(length):
                word = words[j]
                try:
                    E[:, j] = self.word2em[word]
                except KeyError:
                    pass
            X[i, :] = np.sum(E, axis=1)

        return X

    def encode_doc(self, doc, max_allowed_doc_length=None):
        '''
        1つの文のベクトルを返す
        '''
        words = [w.lower() for w in doc.split(' ')]
        max_len = len(words)
        if max_allowed_doc_length is not None:
            max_len = min(len(words), max_allowed_doc_length)
        E = np.zeros(shape=(self.embedding_dim, max_len))
        X = np.zeros(shape=(self.embedding_dim, ))
        for j in range(max_len):
            word = words[j]
            try:
                E[:, j] = self.word2em[word]
            except KeyError:
                pass
        X[:] = np.sum(E, axis=1)
        return X
