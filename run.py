from datapreparation import datapreparation
from token_fit import token_fit
from model.bert_model import bert_model
from model.lstm_model import lstm
import random
import numpy as np
import pandas as pd
import os
import gc
from keras.preprocessing.text import Tokenizer

from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from fastai.train import Learner
import torch
import sys
EMB_PATHS = [
    '../input/fasttext-crawl-300d-2m/crawl-300d-2M.vec',
    '../input/glove840b300dtxt/glove.840B.300d.txt'
]

JIGSAW_PATH = '../input/jigsaw-unintended-bias-in-toxicity-classification/'

package_dir = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"
sys.path.append(package_dir)
debug = False
EMB_MAX_FEAT = 300
MAX_LEN = 220
max_features = 400000
batch_size = 512
NUM_EPOCHS = 4
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 512
NUM_MODELS = 1

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def load_embeddings(path):
    with open(path) as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in f)


def build_embedding_matrix(tok, path):
    word_docs = tok.word_docs
    word_index = tok.word_index
    embeddings_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, EMB_MAX_FEAT))
    vocabs = {key: word_docs[key] for key in word_docs.keys() if word_docs[key] > 0 and key in embeddings_index}
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
def build_embeddings(tok):
    embedding_matrix = np.concatenate(
        [build_embedding_matrix(tok, f) for f in EMB_PATHS], axis=-1)
    return embedding_matrix


def submit(sub_preds, debug):
    submission = pd.read_csv(os.path.join(JIGSAW_PATH, 'sample_submission.csv'), index_col='id')
    if debug:
        submission = submission.iloc[:10000]
    submission['prediction'] = sub_preds
    submission.reset_index(drop=False, inplace=True)
    submission.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    data = datapreparation(debug)
    data.load_data_and_clean()
    train = data.train
    test = data.test
    get_token = token_fit()
    x_train, x_test, y_train, y_train1, y_train2, tok, loss_weight, train_lengths, test_lengths =\
        get_token.token_fit(train, test)
    max_features = min(max_features, len(tok.word_index) + 1)
    bert_model = bert_model()
    preds1 = bert_model.bert_predict(test, "../input/bert-pretrained-models/uncased_l-12_h-768_a-12/uncased_L-12_H-768_A-12/",
                              ["../input/uncasedmodel2/","../input/uncasedmodel1/"])
    preds2 = bert_model.bert_predict(test, "../input/bert-pretrained-models/cased_l-12_h-768_a-12/cased_L-12_H-768_A-12/",
                              ["../input/casedmodel3/","../input/casedmodel4/"])

    del train, test
    gc.collect()
    if debug:
        embedding_matrix = np.zeros((max_features, EMB_MAX_FEAT * 2))
    else:
        from nltk.stem import PorterStemmer

        ps = PorterStemmer()
        from nltk.stem.lancaster import LancasterStemmer

        lc = LancasterStemmer()
        from nltk.stem import SnowballStemmer

        sb = SnowballStemmer("english")
        embedding_matrix = build_embeddings(tok)
        del sb, lc, ps
        gc.collect()
    lstm = lstm(test_lengths, train_lengths,embedding_matrix,loss_weight)
    preds0 = lstm.train_(x_train, y_train2, y_train1, x_test)
    sub_preds = preds0[:, 2] * 0.2 + preds1 * 0.48 + preds2 * 0.32
    submit(sub_preds, debug)
