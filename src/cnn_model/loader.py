from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.utils import to_categorical
from keras.models import load_model
from sklearn.model_selection import train_test_split
from imutils import paths

import numpy as np
import random
import json
import glob
import cv2
import os


class DataLoader():

    @staticmethod
    def get_default_datagen():
        return ImageDataGenerator(
            rotation_range=180,
            width_shift_range=0.2,   # 6 images - 0.2
            height_shift_range=0.3,  # 8 images - 0.3
            zoom_range=[0.7, 1.3],   # 10 images - [0.6, 1.4]
            channel_shift_range=10,  # 10 images - 10
            fill_mode='nearest',
            horizontal_flip=True,    # 5 images - True
            vertical_flip=True,      # 5 images - True
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
        labels_counter = 0
        num_of_images_per_category = {}

        # Create labels_dict
        for category in os.listdir(root_path):
            labels_dict.update({
                category: labels_counter
            })
            labels_counter += 1

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

        return (image_data, image_labels, labels_dict)

    @staticmethod
    def split_data(training_data, labels, split_size=0.25):
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

    @staticmethod
    def load_and_preprocess_image(image_path, width, height):
        # Preprocess image
        image = cv2.resize(
            cv2.imread(image_path),
            (width, height)
        )
        image = img_to_array(image.astype("float") / 255.0)
        return np.expand_dims(image, axis=0)

    @staticmethod
    def load_model_data(model_path):
        # Get model name
        model_name = model_path.split(os.path.sep)[-2]

        # Read data from json file
        json_file = open(model_path + model_name + '.json')
        model_data = json.load(json_file)

        return (
            load_model(model_path + model_name + '.model'),
            model_data['width'],
            model_data['height'],
            model_data['labels'],
        )
