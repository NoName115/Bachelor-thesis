from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import backend as K
from abc import ABCMeta, abstractmethod
from preprocessing import Preprocessor

import json
import os


class Model():

    __metaclass__ = ABCMeta

    def __init__(self, data_shape, labels_dict,
                 model_name=None, model=None):
        """Initalization of CNN model

        data_shape - (n, x, y, 1 or 3), shape of training data
        lables_dict - dict, lables dictionary
        model_name - string, if None name is class name
        model - Sequential, 
        """
        self.width = data_shape[1]
        self.height = data_shape[2]
        self.depth = data_shape[3]
        self.labels_dict = labels_dict
        self.num_of_classes = len(labels_dict)
        self.model_name = model_name if (model_name) else self.__class__.__name__
        self.optimizer = None

        # Initialize first layer shape
        self.input_shape = (self.height, self.width, self.depth)
        if (K.image_data_format() == "channels_first"):
            self.input_shape = (self.depth, self.height, self.width)

        if (not model):
            self.model = self.build()
        else:
            self.model = model

    def get_name(self):
        return self.model_name

    def get_model(self):
        return self.model
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train(self, train_x, train_y, val_x, val_y,
              datagen, epochs, batch_size):
        self.__compile()

        return self.model.fit_generator(
            datagen.flow(train_x, train_y, batch_size=batch_size),
            steps_per_epoch=len(train_x) // batch_size,
            epochs=epochs,
            validation_data=(val_x, val_y),
            validation_steps=len(val_x) // batch_size
        )

    @abstractmethod
    def __compile(self, loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy']):
        self.model.compile(
            loss=loss,
            optimizer=self.optimizer if (self.optimizer) else optimizer,
            metrics=metrics
        )

    @abstractmethod
    def build(self):
        '''Notes

        Convolution layer:
            (W - F + 2P)/S + 1 = cele cislo
            W - velkost vstupnych dat
            F - velkost filtra
            P - padding, zero-padding = 1 ('same')
            S - stride size
        '''
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


class MyModel(Model):

    def build(self):
        # Input shape 64x64x1
        model = Sequential()

        model.add(Conv2D(
            16, (2, 2), strides=(1, 1), padding='same',
            input_shape=self.input_shape, #activation='relu'
            )
        )   # Output 64x64x16
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))   # Output 32x32x16

        model.add(Conv2D(32, (2, 2), strides=(1, 1), padding='same'))   # Output 32x32x32
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))   # Output 16x16x32

        model.add(Conv2D(64, (2, 2), strides=(1, 1), padding='same'))   # Output 16x16x64
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))   # Output 8x8x64

        model.add(Conv2D(128, (2, 2), strides=(1, 1), padding='same'))   # Output 8x8x128
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))   # Output 4x4x128

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))

        #model.add(Dropout(0.5))
        model.add(Dense(self.num_of_classes))
        model.add(Activation('softmax'))

        return model

