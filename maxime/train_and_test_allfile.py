import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Activation, InputLayer, Conv2D, MaxPooling2D, Flatten, \
    GRU, Dropout, Bidirectional
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import save_model
from utils.pltimage import write_seq
from sklearn.metrics import accuracy_score
import numpy as np

import os

from utils.dataset import GobuleRawFullDataSet
import time
import pickle as pk

dir_path = "tmp/train1/"
dir_path_test = "tmp/test1/"
epochs = 500
steps_by_file = 3


def named_logs(model, logs, logs_val, cls1, cls2, cls3):
    result = {}
    for l in zip(model.metrics_names, logs):
        result[l[0]] = l[1]
    for l in zip(model.metrics_names, logs_val):
        result["validation_" + l[0]] = l[1]
    for l in zip(model.metrics_names, cls1):
        result["validation0_" + l[0]] = l[1]
    for l in zip(model.metrics_names, cls2):
        result["validation1_" + l[0]] = l[1]
    for l in zip(model.metrics_names, cls3):
        result["validation2_" + l[0]] = l[1]

    return result


model = Sequential()
model.add(InputLayer(input_shape=(None, 31, 31, 1)))
model.add(TimeDistributed(Conv2D(8, kernel_size=(3, 3),
                                 activation='relu', )))
model.add(TimeDistributed(Conv2D(16, (3, 3), activation='relu')))
model.add(TimeDistributed(Conv2D(24, (3, 3), activation='relu')))

model.add(TimeDistributed(Flatten()))

model.add(Bidirectional(GRU(128, return_sequences=True)))
model.add(Bidirectional(GRU(128)))

model.add(Dense(64, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(3, activation="softmax"))

tbcallback = TensorBoard(log_dir=f'logs/run1-{time.time()}', batch_size=128 * 3, update_freq=0, histogram_freq=0,
                         write_graph=False)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

tbcallback.set_model(model)

files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
test_files = [os.path.join(dir_path_test, f) for f in os.listdir(dir_path_test) if
              os.path.isfile(os.path.join(dir_path_test, f))]

tbcallback.on_train_begin(None)
counter = 0

for e in range(epochs):
    for ef, data_f in enumerate(files):
        dataset = GobuleRawFullDataSet(data_f, test_set=0, validation_set=0.4)

        for batch_id in range(steps_by_file):
            print(f"epoch={e}, file_number={ef}, batch_id={batch_id}")
            X, Y = dataset.getBatch((64, 64, 64))
            logs = model.train_on_batch(X, Y)

            counter += 1

            if counter % 30 == 1 and counter > 1:
                X, Y = dataset.validation(sizes=(128, 0, 0))
                cls1 = model.evaluate(X, Y)

                X, Y = dataset.validation(sizes=(0, 128, 0))
                cls2 = model.evaluate(X, Y)

                X, Y = dataset.validation(sizes=(0, 0, 128))
                cls3 = model.evaluate(X, Y)

                logs_val = [(cls1[i] + cls2[i] + cls3[i]) / 3 for i in range(len(cls1))]

                n_logs = named_logs(model, logs, logs_val, cls1, cls2, cls3)

                print(n_logs, cls1, cls2, cls3)
                tbcallback.on_batch_end(counter, n_logs)



                if logs_val[1] > 0.95 and e > 15:
                    pred_1, pred_2, pred_3 = [], [], []
                    ytrue_1, ytrue_2, ytrue_3 = [], [], []

                    for datatest_f in test_files:
                        dataset_test = GobuleRawFullDataSet(datatest_f, test_set=0, validation_set=0)

                        for X, Y in dataset_test.generate_batches(0, sizes=32):
                            pred_1.append(model.predict(X))
                            ytrue_1.append(Y)


                        for X, Y in dataset_test.generate_batches(1, sizes=32):
                            pred_2.append(model.predict(X))
                            ytrue_2.append(Y)

                        for X, Y in dataset_test.generate_batches(2, sizes=32):
                            pred_3.append(model.predict(X))
                            ytrue_3.append(Y)

                    score_1 = accuracy_score(np.concatenate(ytrue_1), np.argmax(np.concatenate(pred_1), axis=1))
                    score_2 = accuracy_score(np.concatenate(ytrue_2), np.argmax(np.concatenate(pred_2), axis=1))
                    score_3 = accuracy_score(np.concatenate(ytrue_3), np.argmax(np.concatenate(pred_3), axis=1))

                    with open("score.txt", 'w') as f:
                        print((score_1, score_2, score_3), file=f)
                    break


save_model(model, f"tmp/model-{time.time()}.chkpt", "wb")
tbcallback.on_train_end(None)
