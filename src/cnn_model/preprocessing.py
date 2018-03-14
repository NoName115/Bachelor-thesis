from keras.preprocessing.image import ImageDataGenerator
from skimage.color import rgb2grey
from printer import print_warning, print_info

import numpy as np


class Preprocessor():

    def __init__(self):
        self.func_list = []
        self.datagen = None
        self.datagen_args = None

    def add_func(self, func):
        self.func_list.append(func)

    def apply(self, image_data):
        # DEBUG
        print_info('Applying preprocessing...')

        # Image preprocessing
        for func in self.func_list:
            # DEBUG
            print_info(func.__name__, 1)
            image_data = func(image_data)

        return image_data

    def __str__(self):
        return (
            '[Preprocessing] ' +
            ' --> '.join(func.__name__ for func in self.func_list)
        )

    def set_datagen(self, default=True, **kwargs):
        # Set default settings
        if (default):
            datagen_kwargs = dict(
                rotation_range=180,
                width_shift_range=0.15,   # 6 images - 0.2
                height_shift_range=0.15,  # 8 images - 0.3
                #zoom_range=[0.7, 1.3],   # 10 images - [0.6, 1.4]
                #channel_shift_range=10,  # 10 images - 10
                fill_mode='nearest',
                horizontal_flip=True,    # 5 images - True
                vertical_flip=True,      # 5 images - True
            )
        else:
            datagen_kwargs = kwargs

        self.datagen_args = datagen_kwargs
        self.datagen = Preprocessing.get_datagen(**datagen_kwargs)

    def get_datagen(self):
        return self.datagen

    def fit(self, **kwargs):
        self.datagen.fit(**kwargs)

    def get_json(self):
        return {
            'func_list': [func.__name__ for func in self.func_list],
            'datagen_args': self.datagen_args,
        }


class Preprocessing():

    @staticmethod
    def get_datagen(**kwargs):
        return ImageDataGenerator(**kwargs)

    @staticmethod
    def reshape(image_data):
        '''Reshape input grayscale data into (x, y, 1)
        '''
        if (len(image_data.shape) == 3):    # List of images
            return image_data.reshape(image_data.shape + (1,))
        else:
            return image_data.reshape((1,) + image_data.shape + (1,))

    @staticmethod
    def denormalize(image_data):
        '''Multiply input data with 255.0 to [0..255]
        '''
        return image_data * 255.0

    @staticmethod
    def normalize(image_data):
        '''Normalize input data to [0..1] scale
        '''
        return image_data / 255.0

    @staticmethod
    def grayscale(image_data):
        '''Rescale input RGB data to grayscale

        Input images must be normalized & have shape (x, y, 3)
        Return data have shape (x, y)
        '''
        if (len(image_data.shape) == 4):    # List of images
            return np.array([rgb2grey(image) for image in image_data])
        else:
            return rgb2grey(image_data)
