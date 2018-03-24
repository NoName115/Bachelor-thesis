from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import backend as K
from abc import ABCMeta, abstractmethod
from printer import print_info

import numpy as np


class Model():

    __metaclass__ = ABCMeta

    def __init__(self, data_shape, labels_dict, model_type,
                 model_name=None, model_folder='', model=None):
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
        self.model_folder = model_folder
        self.model_type = model_type

        self.batch_size = -1
        self.epochs = -1

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

    def set_model_folder(self, model_folder_path):
        self.model_folder = model_folder_path

    def train(self, train_x, train_y, val_x, val_y,
              datagen=None, epochs=40, batch_size=16,
              loss='binary_crossentropy', optimizer='rmsprop',
              metrics=['accuracy']):
        # Debug
        print_info('Training model...')

        # Save training parameters
        self.batch_size = batch_size
        self.epochs = epochs

        # Compile model
        self.model.compile(
            loss=loss,
            optimizer=optimizer,
            metrics=metrics
        )

        # Train model
        if (datagen):
            self.model.fit_generator(
                datagen.flow(train_x, train_y, batch_size=batch_size),
                steps_per_epoch=int(np.ceil(len(train_x) / batch_size)),
                epochs=epochs,
                validation_data=datagen.flow(val_x, val_y, batch_size=batch_size),
                validation_steps=int(np.ceil(len(val_x) / batch_size))
            )
        else:
            self.model.fit(
                train_x, train_y,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(val_x, val_y)
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

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))

        model.add(Dropout(0.5))
        model.add(Dense(self.num_of_classes))
        model.add(Activation('sigmoid'))

        return model


class LeNet(Model):

    def build(self):
        """Dlho to trva a bez znacnych vysledok
        """
        # Initialize the model
        model = Sequential()

        # first set of CONV => RELU => POOL layers
        model.add(
            Conv2D(20, (5, 5), padding='same', input_shape=self.input_shape
        ))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(Conv2D(50, (5, 5), padding='same'))
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
        model = Sequential()

        model.add(Conv2D(
            16, (2, 2), padding='same',
            input_shape=self.input_shape, activation='relu'
            )
        )   # Output 64x64x16
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))   # Output 32x32x16
        model.add(Dropout(0.20))

        model.add(Conv2D(32, (2, 2), padding='same', activation='relu'))   # Output 32x32x32
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))   # Output 16x16x32
        model.add(Dropout(0.20))

        model.add(Conv2D(64, (2, 2), padding='same', activation='relu'))   # Output 16x16x64
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))   # Output 8x8x64
        model.add(Dropout(0.20))

        model.add(Conv2D(128, (2, 2), padding='same', activation='relu'))   # Output 8x8x128
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))   # Output 4x4x128
        model.add(Dropout(0.20))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))

        model.add(Dense(self.num_of_classes, activation='softmax'))

        return model


class VGG16(Model):

    def build(self):
        """Architecture inspired by VGG16
        """
        # Input shape 224x224x3
        model = Sequential()

        model.add(Conv2D(
            32, (2, 2), padding='same', input_shape=self.input_shape, activation='relu'
        ))
        model.add(Conv2D(32, (2, 2), padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(64, (2, 2), padding='same'))
        model.add(Conv2D(64, (2, 2), padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Conv2D(128, (2, 2), padding='same'))
        model.add(Conv2D(128, (2, 2), padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        #model.add(Dense(1024, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(1024, activation='relu'))
        #model.add(Dropout(0.5))

        model.add(Dense(self.num_of_classes, activation='softmax'))

        return model
