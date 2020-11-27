import json
import numpy as np
from keras import applications
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dropout, Flatten, Dense
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from tqdm.keras import TqdmCallback

import config as conf


top_model_weights_path = conf.top_model_weights_file
top_model_json_path = conf.top_model_json_file
test_feature_file = conf.test_feature_file
nb_test_samples = conf.nb_test_samples
batch_size = conf.batch_size
epochs = conf.epochs

evaluation_json_file = conf.evaluation_json_file


def evaluate():
    test_data = np.load(test_feature_file)
    test_labels = np.array(
        [0] * (nb_test_samples // 2) + [1] * (nb_test_samples // 2))

    json_file = open(top_model_json_path, 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.load_weights(top_model_weights_path)
    result = model.evaluate(test_data, test_labels,
                            batch_size=batch_size,
                            verbose=0)

    json_dict = {}
    json_dict['loss'] = result[0]
    json_dict['accuracy'] = result[1]

    with open(evaluation_json_file, 'w') as jsonFile:
        jsonFile.write(json.dumps(json_dict, indent=4))

evaluate()

