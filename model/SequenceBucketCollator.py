import torch
EMB_MAX_FEAT = 300
MAX_LEN = 220
max_features = 400000
batch_size = 512
NUM_EPOCHS = 4
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 512
NUM_MODELS = 1
class SequenceBucketCollator():
    def __init__(self, choose_length, sequence_index, length_index, label_index=None):
        self.choose_length = choose_length
        self.sequence_index = sequence_index
        self.length_index = length_index
        self.label_index = label_index

    def __call__(self, batch):
        batch = [torch.stack(x) for x in list(zip(*batch))]

        sequences = batch[self.sequence_index]
        lengths = batch[self.length_index]

        length = self.choose_length(lengths)
        mask = torch.arange(start=MAX_LEN, end=0, step=-1) < length
        padded_sequences = sequences[:, mask]

        batch[self.sequence_index] = padded_sequences

        if self.label_index is not None:
            return [x for i, x in enumerate(batch) if i != self.label_index], batch[self.label_index]

        return batch