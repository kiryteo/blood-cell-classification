
from torch.utils.data import Dataset
import pickle as pk
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch

class GlobulesDataset(Dataset):
    def __init__(self, path):
        with open(path, "rb") as f:
            self.x, self.y = pk.load(f, encoding="bytes")

        values, counts = np.unique(self.y, return_counts=True)

        counts = 1 / counts

        self.freqs = np.array((counts[0]*10, counts[1]*10, counts[2]/10000))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        return self.x[item] / 255, self.y[item]


def max_length(x):
    l = 0

    for s in x:
        ltmp = len(s)
        if ltmp > l:
            l = ltmp
    return l

def pad_collate(batch):
    xx, yy = zip(*batch)

    l = max_length(xx)

    X = np.zeros((l, len(batch), 31,31))

    for i, x in enumerate(xx):
        X[:len(x), i] = x

    yy = np.array(yy)

    return X, torch.Tensor(yy).long()