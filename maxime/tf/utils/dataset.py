import pickle as pk
import numpy as np


def split_data(N, validation_set, test_set, seed=896):
    np.random.seed(seed)
    indices = np.arange(N)
    # np.random.shuffle(indices)

    idxTest = indices[:int(N * test_set)]
    indices = indices[int(N * test_set):]

    idxValidation = indices[:int(N * validation_set)]
    idxTrain = indices[int(N * validation_set):]

    return idxTrain, idxValidation, idxTest


class GobuleRawFullDataSet:
    def __init__(self, path, validation_set=0.2, test_set=0):
        self.data = pk.load(open(path, "rb"), encoding="bytes")

        self.idxTrain = []
        self.idxVal = []
        self.idxTest = []

        for i in range(3):
            _idxTrain, _idxValidation, _idxTest = split_data(len(self.data[i]), validation_set, test_set)

            self.idxTrain.append(_idxTrain)
            self.idxVal.append(_idxValidation)
            self.idxTest.append(_idxTest)

    def generate_batches(self, cls_id, sizes=250):

        s = len(self.idxTrain[cls_id]) // sizes
        if s > 0:
            data = np.array_split(self.idxTrain[cls_id], s)
        else:
            data = [self.idxTrain[cls_id]]

        for ids in data:
            size_max = self.max_len(cls_id, ids)

            X = np.zeros((len(ids), size_max, 31, 31))
            Y = np.zeros(len(ids))

            for i, a in enumerate(ids):
                X[i, :len(self.data[cls_id][a])] = np.array(self.data[cls_id][a])
                Y[i] = cls_id

            yield np.reshape(X, (*X.shape, 1)) / 255, Y

    def pad_batch(self, A, B, C, sizes):
        size_max = max(self.max_len(0, A), self.max_len(1, B), self.max_len(2, C))

        X = np.zeros((sum(sizes), size_max, 31, 31))
        Y = np.zeros(sum(sizes))

        for i, a in enumerate(A):
            X[i, :len(self.data[0][a])] = np.array(self.data[0][a])
            Y[i] = 0

        for i, b in enumerate(B):
            X[sizes[0] + i, :len(self.data[1][b])] = np.array(self.data[1][b])
            Y[sizes[0] + i] = 1

        for i, c in enumerate(C):
            X[sizes[0] + sizes[1] + i, :len(self.data[2][c])] = np.array(self.data[2][c])
            Y[sizes[0] + sizes[1] + i] = 2

        return np.reshape(X, (*X.shape, 1)) / 255, Y

    def getBatch(self, sizes=(250, 250, 250)):
        A = np.random.choice(self.idxTrain[0], sizes[0])
        B = np.random.choice(self.idxTrain[1], sizes[1])
        C = np.random.choice(self.idxTrain[2], sizes[2])

        return self.pad_batch(A, B, C, sizes)

    def validation(self, sizes=(128, 128, 128)):
        A = np.random.choice(self.idxVal[0], sizes[0])
        B = np.random.choice(self.idxVal[1], sizes[1])
        C = np.random.choice(self.idxVal[2], sizes[2])

        size_max = max(self.max_len(0, A), self.max_len(1, B), self.max_len(2, C))

        sizes = (len(A), len(B), len(C))

        X = np.zeros((sum(sizes), size_max, 31, 31))
        Y = np.zeros(sum(sizes))

        for i, a in enumerate(A):
            X[i, :len(self.data[0][a])] = np.array(self.data[0][a])
            Y[i] = 0

        for i, b in enumerate(B):
            X[sizes[0] + i, :len(self.data[1][b])] = np.array(self.data[1][b])
            Y[sizes[0] + i] = 1

        for i, c in enumerate(C):
            X[sizes[0] + sizes[1] + i, :len(self.data[2][c])] = np.array(self.data[2][c])
            Y[sizes[0] + sizes[1] + i] = 2

        return np.reshape(X, (*X.shape, 1)) / 255, Y

    def get_valcls(self, id):

        size_max = self.max_len(0, self.idxVal[id])

        X = np.zeros((len(self.idxVal[id]), size_max, 31, 31))
        Y = np.zeros(len(self.idxVal[id]))

        for i, a in enumerate(self.idxVal[id]):
            X[i, :len(self.data[0][a])] = np.array(self.data[0][a])
            Y[i] = 0

    def max_len(self, i, A):
        m = 0

        for a in A:
            l = len(self.data[i][a])
            if m < l:
                m = l

        return m
