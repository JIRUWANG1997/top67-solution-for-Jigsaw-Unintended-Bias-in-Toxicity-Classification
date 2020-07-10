import numpy as np
from keras.preprocessing.text import Tokenizer
max_features = 400000
import torch
MAX_LEN = 220
from keras.preprocessing.sequence import pad_sequences
class token_fit():

    def token_fit(self, train, test):
        identity_columns = ['asian', 'atheist', 'bisexual', 'black', 'buddhist', 'christian', 'female', 'heterosexual',
                            'hindu',
                            'homosexual_gay_or_lesbian', 'intellectual_or_learning_disability', 'jewish', 'latino', 'male',
                            'muslim', 'other_disability',
                            'other_gender', 'other_race_or_ethnicity', 'other_religion', 'other_sexual_orientation',
                            'physical_disability', 'psychiatric_or_mental_illness', 'transgender', 'white']
        # Overall
        weights = np.ones((len(train),)) / 4
        # Subgroup
        weights += (train[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int) / 4
        # Background Positive, Subgroup Negative
        weights += (((train['target'].values >= 0.5).astype(bool).astype(np.int) +
                     (train[identity_columns].fillna(0).values < 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
            bool).astype(np.int) / 4
        # Background Negative, Subgroup Positive
        weights += (((train['target'].values < 0.5).astype(bool).astype(np.int) +
                     (train[identity_columns].fillna(0).values >= 0.5).sum(axis=1).astype(bool).astype(np.int)) > 1).astype(
            bool).astype(np.int) / 4
        loss_weight = 1.0 / weights.mean()
        tok = Tokenizer(num_words=max_features, filters="", lower=False)
        tok.fit_on_texts(list(train.comment_text) + list(test.comment_text))
        x_train = tok.texts_to_sequences(train.comment_text.values)
        train_lengths = torch.from_numpy(np.array([len(x) for x in x_train]))
        x_train = torch.from_numpy(pad_sequences(x_train, maxlen=MAX_LEN))
        x_test = tok.texts_to_sequences(test.comment_text.values)
        test_lengths = torch.from_numpy(np.array([len(x) for x in x_test]))
        x_test = torch.from_numpy(pad_sequences(x_test, maxlen=MAX_LEN))
        y_train = np.vstack([(train['target'].values >= 0.5).astype(np.int), weights]).T
        y_train1 = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']].values
        y_train2 = np.vstack([(train.target.values >= i).astype(int) for i in [0.01, 0.25, 0.5, 0.75]])
        y_train2 = np.vstack([y_train2, weights]).T
        return x_train, x_test, y_train, y_train1, y_train2, tok, loss_weight, train_lengths, test_lengths
