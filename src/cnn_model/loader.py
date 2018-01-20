from keras.preprocessing.image import ImageDataGenerator, img_to_array
from sklearn.model_selection import train_test_split
from imutils import paths

import numpy as np
import random
import glob
import cv2
import os


class DataLoader():

    @staticmethod
    def get_default_datagen():
        return ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            #fill_mode="nearest"
        )

    @staticmethod
    def get_rescale_datagen():
        return ImageDataGenerator(
            rescale=1./255
        )

    @staticmethod
    def load_data_with_datagen(dir_path, width, height, datagen, batch_size):
        return datagen.flow_from_directory(
            dir_path,
            target_size=(width, height),
            batch_size=batch_size,
            class_mode='binary'
        )

    @staticmethod
    def load_scaled_data_with_labels(
        root_path, width, height, correct_dataset_size=True
    ):
        image_data = []
        image_labels = []
        labels_dict = {}
        number_of_labels = 0
        num_of_images_per_category = {}

        # Create labels_dict
        for category in os.listdir(root_path):
            labels_dict.update({
                category: number_of_labels
            })
            number_of_labels += 1

        # Correct dataset size
        category_counter = []
        max_images = None
        if (correct_dataset_size):
            for cat_path in glob.glob(root_path + '/*'):
                category_counter.append(
                    len(os.listdir(cat_path))
                )
            max_images = min(category_counter)

        # Grab the image paths and randomly shuffle them
        image_paths = list(paths.list_images(root_path))
        random.seed(42)
        random.shuffle(image_paths)

        for path in image_paths:
            # Load image label
            label = path.split(os.path.sep)[-2]

            # Check number of images per category
            if (not label in num_of_images_per_category):
                num_of_images_per_category.update({
                    label: 0
                })
            
            # Correct dataset size
            if (correct_dataset_size and num_of_images_per_category[label] >= max_images):
                break

            # Load and rescale images
            image = cv2.resize(cv2.imread(path), (width, height))
            image_data.append(img_to_array(image))

            # Labels operations
            image_labels.append(labels_dict[label])
            num_of_images_per_category[label] += 1


        # Scale the raw pixel intensities to the range [0, 1]
        image_data = np.array(image_data, dtype="float") / 255.0
        image_labels = np.array(image_labels)

        return (image_data, image_labels, labels_dict, number_of_labels)

    @staticmethod
    def split_data(training_data, labels, split_size):
        if (not 0 <= split_size <= 0.5):
            print('[ERROR]: Invalid split size: ' + str(split_size))
            return None

        train_x, test_x, train_y, test_y = train_test_split(
            training_data,
            labels,
            test_size=split_size,
            random_state=42
        )

        return (
            train_x,
            to_categorical(train_y, num_classes=2),
            test_x,
            to_categorical(test_y, num_classes=2)
        )
