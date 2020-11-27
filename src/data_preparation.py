'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import applications

import config as conf

train_data_dir = conf.train_data_path
validation_data_dir = conf.validation_data_path
test_data_dir = conf.test_data_path
train_feature_path = conf.train_feature_file
valid_feature_path = conf.valid_feature_file
test_feature_path = conf.test_feature_file

# dimensions of our images.
img_width, img_height = conf.img_width, conf.img_height

nb_train_samples = conf.nb_train_samples
nb_validation_samples = conf.nb_validation_samples
nb_test_samples = conf.nb_test_samples
batch_size = conf.batch_size

## dimensions of our images.
#img_width, img_height = 150, 150
#
#train_data_dir = 'data/train'
#validation_data_dir = 'data/validation'
#test_data_dir = 'data/test'
#train_feature_path = './model/feature/bottleneck_features_train.npy'
#valid_feature_path = './model/feature/bottleneck_features_validation.npy'
#test_feature_path = './model/feature/bottleneck_features_test.npy'
#nb_train_samples = 1000
#nb_validation_samples = 800
#nb_test_samples = 200
#batch_size = 10


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(train_feature_path, bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(valid_feature_path, bottleneck_features_validation)

    generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_test = model.predict_generator(
        generator, nb_test_samples // batch_size)
    np.save(test_feature_path, bottleneck_features_test)


save_bottlebeck_features()
