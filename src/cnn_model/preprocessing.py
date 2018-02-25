from keras.preprocessing.image import ImageDataGenerator


class Preprocessor():

    @staticmethod
    def get_datagen(zca_whitening=False):
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
