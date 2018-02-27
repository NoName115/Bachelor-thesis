from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2grey
from printer import print_warning, print_info

import numpy as np


class Preprocessor():

    def __init__(self):
        self.func_list = []
        self.data_gen = None

    def add_func(self, func):
        self.func_list.append(func)

    def set_datagen(self, data_gen):
        if (not self.data_gen):
            self.data_gen = data_gen
        else:
            print_warning("Data generator already added")

    def apply(self, image_data):
        # Debug
        print_info('Applying preprocessing...')
        # Image preprocessing
        for func in self.func_list:
            print_info(func.__name__, 1)    # DEBUG
            image_data = func(image_data)

        return image_data

    def __str__(self):
        return (
            '[Preprocessing] ' +
            ' --> '.join(func.__name__ for func in self.func_list)
        )

    def get_json(self):
        pass


class Preprocessing():

    @staticmethod
    def get_default_datagen(zca_whitening=False):
        return ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.2,   # 6 images - 0.2
            height_shift_range=0.3,  # 8 images - 0.3
            zoom_range=[0.7, 1.3],   # 10 images - [0.6, 1.4]
            channel_shift_range=10,  # 10 images - 10
            fill_mode='nearest',
            horizontal_flip=True,    # 5 images - True
            vertical_flip=True,      # 5 images - True
            zca_whitening=zca_whitening
        )

    @staticmethod
    def normalize(image_data):
        return image_data / 255.0

    @staticmethod
    def grayscale(image_data):
        return np.array([rgb2grey(image) for image in image_data])
