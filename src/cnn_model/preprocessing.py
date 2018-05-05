from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from skimage.color import rgb2grey
from sklearn.utils import shuffle
from imutils import rotate_bound
from printer import print_info, print_warning, print_error
from cv2 import resize

import numpy as np
import math


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

    @staticmethod
    def flat(image_data):
        """Flat input data into shape (num_of_samples, num_of_features)
        """
        return image_data.reshape((len(image_data), -1))

    @staticmethod
    def get_correct_angle_label(rand_angle, labels_dict):
        label = round(rand_angle / (360 / len(labels_dict)))
        # Case when 359 / 5 = 72, array max indx is 71
        if (label == len(labels_dict)):
            label = 0
        return label

    @staticmethod
    def rotate_yaw_image(image, label, labels_dict):
        angle = label * (360 / len(labels_dict))

        # Horizontal flip, no effect on angle
        if (np.random.random() < 0.5):
            image = np.flip(image, 0)

        # Vertical flip
        if (np.random.random() < 0.5):
            image = np.flip(image, 1)
            if (270 <= angle <= 359):
                angle = abs(angle - 540)
            elif (0 <= angle <= 90):
                angle = abs(angle - 180)
            else:
                print_warning("Invalid angle " + str(angle))

        new_label = Preprocessing.get_correct_angle_label(angle, labels_dict)
        return image, angle, new_label

    @staticmethod
    def rotate_pitch_image(image, label, labels_dict):
        angle = label * (360 / len(labels_dict))

        # Horizontal flip
        if (np.random.random() < 0.5):
            image = np.flip(image, 0)
            if (270 <= angle <= 359):
                angle = 270 - angle + 270
            elif (0 <= angle <= 90):
                angle = abs(angle - 180)
            else:
                print_warning("Invalid angle " + str(angle))

        # Vertical flip, no effect on angle
        if (np.random.random() < 0.5):
            image = np.flip(image, 1)

        new_label = Preprocessing.get_correct_angle_label(angle, labels_dict)
        return image, angle, new_label

    @staticmethod
    def rotate_and_crop_image(image, labels_dict):
        """Rotate input image by random range & crop black egdes

        Return rotated_image, angle, label
        """
        rand_angle = np.random.randint(0, 360)

        # Horizonal flip
        if (np.random.random() < 0.5):
            image = np.flip(image, 0)

        # WARNING
        # Image is rotated to minus angle
        rotated = Preprocessing.__crop_rotated_image(
            rotate_bound(image, rand_angle),
            rand_angle,
            image.shape[0],
            image.shape[1]
        )

        rand_angle = 360 - rand_angle
        label = Preprocessing.get_correct_angle_label(
            rand_angle, labels_dict
        )

        return rotated, rand_angle, label

    @staticmethod
    def __crop_rotated_image(image, angle, height, width):
        return resize(
            Preprocessing.__crop_around_center(
                image,
                *Preprocessing.__largest_rotated_rect(
                    width,
                    height,
                    math.radians(angle)
                )
            ),
            (width, height)
        )

    @staticmethod
    def __crop_around_center(image, width, height):
        """
        Crops it to the given width and height, around it's centre point
        Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
        """
        image_size = (image.shape[1], image.shape[0])
        image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

        if(width > image_size[0]):
            width = image_size[0]

        if(height > image_size[1]):
            height = image_size[1]

        x1 = int(image_center[0] - width * 0.5)
        x2 = int(image_center[0] + width * 0.5)
        y1 = int(image_center[1] - height * 0.5)
        y2 = int(image_center[1] + height * 0.5)

        return image[y1:y2, x1:x2]

    @staticmethod
    def __largest_rotated_rect(w, h, angle):
        """
        Given a rectangle of size w x h that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle within the rotated rectangle.
        Source: http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
        """
        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)
        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return (
            bb_w - 2 * x,
            bb_h - 2 * y
        )


class AngleGenerator():

    def __init__(self, labels_dict, rotation_type):
        self.angle_range = list(labels_dict.keys())
        self.labels_dict = labels_dict
        self.rotation_type = rotation_type

    def flow(self, set_x, set_y, batch_size):
        while True:
            batch_x = []
            batch_y = []
            set_x, set_y = shuffle(set_x, set_y, random_state=42)

            #np.random.shuffle(set_x)
            for i in range(0, batch_size):
                random_indx = np.random.randint(0, len(set_x))
                image = set_x[random_indx]
                label = np.argmax(set_y[random_indx])

                if (self.rotation_type == 'roll'):
                    rotated, angle, label = Preprocessing.rotate_and_crop_image(
                        image, self.labels_dict
                    )
                elif (self.rotation_type == 'pitch'):
                    rotated, angle, label = Preprocessing.rotate_pitch_image(
                        image, label, self.labels_dict
                    )
                elif (self.rotation_type == 'yaw'):
                    rotated, angle, lable = Preprocessing.rotate_yaw_image(
                        image, label, self.labels_dict
                    )
                else:
                    print_error("Invalid rotation type " + str(rotation_type))

                batch_x.append(rotated)
                batch_y.append(label)

            batch_x = np.array(batch_x, dtype='float32')
            batch_y = to_categorical(
                np.array(batch_y),
                num_classes=len(self.labels_dict)
            )

            if (len(batch_x.shape) == 3):   # Grayscale data
                batch_x = batch_x.reshape(batch_x.shape + (1,))

            yield batch_x, batch_y


class ClassificationGenerator():

    def __init__(self, data_x, data_y, data_p, batch_size, datagen):
        self.data_x = data_x
        self.data_y = data_y
        self.data_p = data_p
        self.batch_size = batch_size
        self.datagen = datagen

    def flow(self):
        print_info('Image augmentation...')

        self.data_x, self.data_y, self.data_p = shuffle(
            self.data_x, self.data_y, self.data_p,
            random_state=42
        )

        batch_x = []
        batch_y = []
        batch_p = []

        for image, label, path in zip(self.data_x, self.data_y, self.data_p):
            image = image.reshape((1,) + image.shape)
            for i in range(0, self.batch_size):
                batch = self.datagen.flow(image, batch_size=1)[0]
                batch_x.append(batch.reshape((
                    batch.shape[1],
                    batch.shape[2],
                    batch.shape[3]
                )))
                batch_y.append(label)
                batch_p.append(path)

        batch_x = np.array(batch_x)
        batch_y = np.array(batch_y)
        batch_p = np.array(batch_p)

        print_info(
            'Input: ' + str(self.data_x.shape) +
            ' Output: ' + str(batch_x.shape),
            1
        )

        return batch_x, batch_y, batch_p
