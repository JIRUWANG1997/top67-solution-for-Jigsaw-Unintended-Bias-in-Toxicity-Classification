import random
import numpy as np
import pandas as pd
import os
import gc
import re
from torch import nn
from keras.preprocessing.text import Tokenizer

from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, CuDNNGRU

from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from fastai.train import Learner
from fastai.train import DataBunch
from fastai.callbacks import TrainingPhase, GeneralScheduler
import torch
from torch.utils import data
from torch.nn import functional as F
import sys
import json
EMB_MAX_FEAT = 300
MAX_LEN = 220
max_features = 400000
batch_size = 512
NUM_EPOCHS = 4
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 512
NUM_MODELS = 1
package_dir = "../input/ppbert/pytorch-pretrained-bert/pytorch-pretrained-BERT"
sys.path.append(package_dir)
from pytorch_pretrained_bert import BertTokenizer, BertForSequenceClassification, BertAdam
from pytorch_pretrained_bert import BertConfig
from SequenceBucketCollator import SequenceBucketCollator
from NeuralNet import NeuralNet
class lstm():
    def __init__(self,test_lengths,train_lengths,embedding_matrix,loss_weight):
        self.test_lengths = test_lengths
        self.train_lengths = train_lengths
        self.embedding_matrix = embedding_matrix
        self.loss_weight = loss_weight

    def custom_loss(self,data, targets):
        ''' Define custom loss function for weighted BCE on 'target' column '''
        bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:, 1:2])(data[:, :1], targets[:, :1])
        bce_loss_2 = nn.BCEWithLogitsLoss()(data[:, 1:], targets[:, 2:])
        return (bce_loss_1 * self.loss_weight) + bce_loss_2

    def custom_loss1(self,data, targets):
        ''' Define custom loss function for weighted BCE on 'target' column '''
        bce_loss_1 = nn.BCEWithLogitsLoss(weight=targets[:, 4:5])(data[:, :4], targets[:, :4])
        bce_loss_2 = nn.BCEWithLogitsLoss()(data[:, 4:], targets[:, 5:])
        return (bce_loss_1 * self.loss_weight) + bce_loss_2
    def seed_everything(self,seed=1234):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
    def train_model(self,learn, test, output_dim, lr=0.001,
                    batch_size=512, n_epochs=5,
                    enable_checkpoint_ensemble=True):
        all_test_preds = []
        checkpoint_weights = [1, 2, 4, 8, 8]
        test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
        n = len(learn.data.train_dl)
        phases = [(TrainingPhase(n).schedule_hp('lr', lr * (0.6 ** (i)))) for i in range(n_epochs)]
        sched = GeneralScheduler(learn, phases)
        learn.callbacks.append(sched)
        for epoch in range(n_epochs):
            learn.fit(1)
            test_preds = np.zeros((len(test), output_dim))
            for i, x_batch in enumerate(test_loader):
                X = x_batch[0].cuda()
                y_pred = self.sigmoid(learn.model(X).detach().cpu().numpy())
                test_preds[i * batch_size:(i + 1) * batch_size, :] = y_pred
            all_test_preds.append(test_preds)

        if enable_checkpoint_ensemble:
            test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)
        else:
            test_preds = all_test_preds[-1]
        return test_preds
    def train_(self,x_train, y_train, y_aux_train, x_test):
        y_train_torch = torch.tensor(np.hstack([y_train, y_aux_train]), dtype=torch.float32)
        test_dataset = data.TensorDataset(x_test, self.test_lengths)
        train_dataset = data.TensorDataset(x_train, self.train_lengths, y_train_torch)
        valid_dataset = data.Subset(train_dataset, indices=[0, 1])
        del x_train, x_test
        gc.collect()

        train_collator = SequenceBucketCollator(lambda lenghts: lenghts.max(),
                                                sequence_index=0,
                                                length_index=1,
                                                label_index=2)

        train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_collator)
        valid_loader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=train_collator)

        databunch = DataBunch(train_dl=train_loader, valid_dl=valid_loader, collate_fn=train_collator)

        del train_dataset, valid_dataset
        gc.collect()

        for model_idx in range(NUM_MODELS):
            all_test_preds = []
            print('Model ', model_idx)
            self.seed_everything(1234 + model_idx)
            model = NeuralNet(self.embedding_matrix, y_aux_train.shape[-1], y_train.shape[-1] - 1)
            if y_train.shape[-1] > 2:
                learn = Learner(databunch, model, loss_func=self.custom_loss1)
            else:
                learn = Learner(databunch, model, loss_func=self.custom_loss)
            test_preds = self.train_model(learn, test_dataset, output_dim=y_train.shape[-1] + y_aux_train.shape[-1] - 1)
            all_test_preds.append(test_preds)
        preds = np.mean(all_test_preds, axis=0)
        return preds
