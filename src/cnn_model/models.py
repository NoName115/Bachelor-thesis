from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import backend as K
from abc import ABCMeta, abstractmethod

import json
import os


class Model():

    __metaclass__ = ABCMeta

    def __init__(self, width, height, labels_dict, depth=3):
        self.width = width
        self.height = height
        self.labels_dict = labels_dict
        self.num_of_classes = len(labels_dict)
        self.depth = depth

        # Initialize first layer shape
        self.input_shape = (self.height, self.width, self.depth)
        if (K.image_data_format() == "channels_first"):
            self.input_shape = (self.depth, self.height, self.width)

        self.model = self.build()

    def save(self, path_to_models, model_name=None):
        if (not model_name):
            model_name = self.__class__.__name__

        # Model folder path
        model_folder_path = path_to_models + model_name
        if (not os.path.exists(model_folder_path)):
            os.mkdir(model_folder_path)

        model_path = model_folder_path + '/' + model_name

        # Save model info
        json_file = open(model_path + '.json', 'w')
        json_file.write(
            json.dumps(
                {
                    'model_name': model_name,
                    'width': self.width,
                    'height': self.height,
                    'labels': self.labels_dict,
                },
                sort_keys=False,
                indent=4,
                separators=(',', ':')
            )
        )
        json_file.close()

        # Save model
        self.model.save(model_path + '.model')

    @abstractmethod
    def build(self):
        pass


class KerasBlog(Model):

    def build(self):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # the model so far outputs 3D feature maps (height, width, features)
        model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dropout(0.5))
        model.add(Dense(self.num_of_classes))
        model.add(Activation('sigmoid'))

        return model


class LeNet(Model):

    def build(self):
        # Initialize the model
        model = Sequential()

        # first set of CONV => RELU => POOL layers
        model.add(
            Conv2D(
                20,
                (5, 5),
                padding='same',
                input_shape=self.input_shape
        ))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(
            Conv2D(
                50,
                (5, 5),
                padding='same'
            )
        )
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        # softmax classifier
        model.add(Dense(self.num_of_classes))
        model.add(Activation("softmax"))

        # return the network architecture
        return model
