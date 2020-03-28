import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Activation, InputLayer, Conv2D, MaxPooling2D, Flatten, GRU, Dropout
from tensorflow.keras.callbacks import TensorBoard

from utils.dataset import GobuleRawFullDataSet
import time

model = Sequential()
model.add(InputLayer(input_shape=(None, 31, 31, 1)))
model.add(TimeDistributed(Conv2D(32, kernel_size=(2, 2),
                                 activation='relu', )))
model.add(TimeDistributed(Conv2D(64, (2, 2), activation='relu')))
model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2))))

model.add(TimeDistributed(Flatten()))

model.add(GRU(128, return_sequences=True))
model.add(GRU(128))

model.add(Dense(64, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(3, activation="softmax"))

tbcallback = TensorBoard(log_dir=f'logs/{time.time()}', batch_size=128*3, update_freq=0, histogram_freq=0, write_graph=False)
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

dataset = GobuleRawFullDataSet("tmp/rawdata.dat", test_set=0, validation_set=0.2)

tbcallback.set_model(model)

def named_logs(model, logs, logs_val):
    result = {}
    for l in zip(model.metrics_names, logs):
        result[l[0]] = l[1]
    for l in zip(model.metrics_names, logs_val):
        result["validation_" + l[0]] = l[1]
    return result

print(len(dataset.data[0]))
print(len(dataset.data[1]))
print(len(dataset.data[2]))

tbcallback.on_train_begin(None)
for batch_id in range(100000):
    print(batch_id)
    X, Y = dataset.getBatch((64, 64, 64))
    logs = model.train_on_batch(X, Y)

    X,Y = dataset.validation(sizes=(128,128,128))

    logs_val = model.evaluate(X, Y)

    n_logs = named_logs(model, logs, logs_val)

    print(n_logs)
    tbcallback.on_batch_end(batch_id, n_logs)


tbcallback.on_train_end(None)
