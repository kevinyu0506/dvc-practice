import numpy as np
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import CSVLogger
from tqdm.keras import TqdmCallback

import config as conf


top_model_weights_path = conf.top_model_weights_file
top_model_json_path = conf.top_model_json_file
train_feature_path = conf.train_feature_file
valid_feature_path = conf.valid_feature_file
nb_train_samples = conf.nb_train_samples
nb_validation_samples = conf.nb_validation_samples
metrics_file = conf.metrics_file

epochs = conf.epochs
batch_size = conf.batch_size


def train_top_model():
    train_data = np.load(train_feature_path)
    train_labels = np.array(
        [0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    validation_data = np.load(valid_feature_path)
    validation_labels = np.array(
        [0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              verbose=0,
              callbacks=[TqdmCallback(), CSVLogger(metrics_file)])
    model.save_weights(top_model_weights_path)

    model_json = model.to_json()

    with open(top_model_json_path, "w") as json_file:
        json_file.write(model_json)


train_top_model()
