from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical, print_summary
from keras.models import load_model, model_from_json
from sklearn.model_selection import train_test_split
from imutils import paths, rotate_bound, rotate
from datetime import datetime
from hashlib import sha1
from printer import print_info, print_warning, print_error, print_blank
from preprocessing import Preprocessor, Preprocessing
from models import Model

import numpy as np
import random
import json
import glob
import cv2
import os


MODEL_SETTINGS_FILE = 'settings.json'
MODEL_BINARY_FILE = 'model.h5'
MODEL_ARCHITECTURE_FILE = 'architecture.json'
MODEL_SUMMARY_FILE = 'summary.txt'

class DataSaver():

    @staticmethod
    def save_image(image, path, image_name, rescale=True):
        '''Save input array as image, only for PNG image extention
        '''
        image = image * 255 if (rescale) else image
        cv2.imwrite(
            path + image_name + ".png",
            image
        )

    @staticmethod
    def save_images(image_data, folder_path, rescale=True, print_mod=5):
        print_info("Saving images to: " + folder_path)

        # Create folder
        if (not os.path.exists(folder_path)):
            os.mkdir(folder_path)

        # Saving images
        for i, image in enumerate(image_data):
            name = str(i)
            DataSaver.save_image(
                image,
                folder_path,
                name,
                rescale
            )
            # Debug
            if (i % print_mod == 0):
                print_info('Saving ' + name + ".png", 1)

    @staticmethod
    def save_image_batch(img_data, datagen, folder_path,
        batch_size=1, loops=15, rescale=True, print_mod=5):
        '''Augment data with input datagen

        Input data must have shape (x, y, 1 or 3)
        '''
        # Debug
        print_info("Saving image batch to: " + folder_path)

        if (len(img_data.shape) == 3):
            img_data = img_data.reshape((1,) + img_data.shape)

        i = 0
        for batch in datagen.flow(img_data, batch_size=batch_size,
             save_to_dir=folder_path, save_prefix='weapon', save_format='jpg'):
            # Debug
            if (i % print_mod == 0):
                print_info("Saving loop " + str(i), 1)
            i += 1
            if i > loops:
                break

    @staticmethod
    def save_model(save_to_folder, model_class, preprocesor, with_datetime=False):
        model_name = model_class.get_name()

        # Model folder path
        model_folder_path = save_to_folder + model_name
        if (with_datetime):
            model_folder_path += '_' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        model_folder_path += "/"

        # Create model folder
        if (not os.path.exists(model_folder_path)):
            os.mkdir(model_folder_path)

        # Add model folder path to model_class
        model_class.set_model_folder(model_folder_path)

        # Debug info
        print_info(
            "Saving model - " + model_name + " to: " + model_folder_path
        )

        # Save model info
        json_file = open(model_folder_path + MODEL_SETTINGS_FILE, 'w')
        json_file.write(
            json.dumps(
                {
                    'model_name': model_name,
                    'width': model_class.width,
                    'height': model_class.height,
                    'labels': model_class.labels_dict,
                    'depth': model_class.depth,
                    'batch_size': model_class.batch_size,
                    'epochs': model_class.epochs,
                    'preprocessing': preprocesor.get_json(),
                    'optimizer': model_class.get_optimizer_as_string()
                },
                sort_keys=False,
                indent=4,
                separators=(',', ':')
            )
        )
        json_file.close()

        # Save model as h5
        model_class.model.save(model_folder_path + MODEL_BINARY_FILE)

        # Save model summary info
        summary_file = open(model_folder_path + MODEL_SUMMARY_FILE, 'w')
        print_summary(
            model_class.model,
            line_length=79,
            positions=None,
            print_fn=lambda in_str: summary_file.write(in_str + '\n')
        )
        summary_file.close()

        # Save model architecture as json
        arch_file = open(model_folder_path + MODEL_ARCHITECTURE_FILE, 'w')
        arch_file.write(
            json.dumps(
                json.loads(model_class.model.to_json()),
                sort_keys=False,
                indent=4,
                separators=(',', ':')
            )
        )
        arch_file.close()

        # Return path where model was saved
        return model_folder_path


class DataLoader():

    @staticmethod
    def load_data_with_datagen(dir_path, width, height, datagen, batch_size):
        return datagen.flow_from_directory(
            dir_path,
            target_size=(width, height),
            batch_size=batch_size,
            class_mode='binary'
        )

    @staticmethod
    def load_scaled_data_from_file(
        file_path, width, height, labels_dict
    ):
        path_list = []

        with open(file_path, 'r') as image_file:
            path_list = [line.rstrip('\n') for line in image_file]

        return DataLoader.__load_images_by_path(
            path_list, labels_dict, width, height
        )

    @staticmethod
    def load_scaled_data_with_labels(
        root_path, width, height,
        labels_dict={}, correct_dataset_size=True,
    ):
        labels_counter = 0
        num_of_images_per_category = {}


        # Create labels_dict
        labels_dict_in = {}
        if (not labels_dict):
            # DEBUG
            print_info("Creating category dictionary...")

            for category in os.listdir(root_path):
                labels_dict_in.update({
                    category: labels_counter
                })
                labels_counter += 1
        else:
            labels_dict_in = labels_dict

        # DEBUG
        print_info("Categories: " + str(labels_dict_in), 1)
        print_info("Loading input dataset...")

        # Correct dataset size
        category_counter = []
        max_images = 0
        if (correct_dataset_size):
            for cat_path in glob.glob(root_path + '/*'):
                category_counter.append(
                    len(os.listdir(cat_path))
                )
            max_images = min(category_counter)

        # Grab the image paths and randomly shuffle them
        image_paths = list(paths.list_images(root_path))

        return DataLoader.__load_images_by_path(
            image_paths, labels_dict_in, width, height,
            correct_dataset_size=correct_dataset_size,
            max_images=max_images
        )

    @staticmethod
    def __load_images_by_path(
        image_paths, labels_dict, width, height,
        correct_dataset_size=False, max_images=0
    ):
        # Initialize data variables
        num_of_images_per_category = dict(
            (label, 0) for label in labels_dict
        )
        image_data = []
        image_labels = []
        image_hash_table = {}
        path_list = []

        for path in image_paths:
            # Load image label
            label = path.split(os.path.sep)[-2]

            # Correct dataset size
            if (correct_dataset_size and num_of_images_per_category[label] >= max_images):
                continue

            # Load and rescale images
            try:
                image = cv2.resize(cv2.imread(path), (width, height))
                image = img_to_array(image)
                image_data.append(image)
            except:
                print_error('Invalid image: ' + image_path)
                continue

            # Labels operations
            image_labels.append(labels_dict[label])
            num_of_images_per_category[label] += 1

            # Add path of loaded image
            path_list.append(path)

            # Add image to image hash table
            image_hash = sha1(image).hexdigest()

            if (image_hash in image_hash_table):
                print_warning('Duplicite image: ', 2)
                print_blank(path, 4)
                print_blank(image_hash_table[image_hash], 4)
                continue

            image_hash_table.update({
                image_hash: path
            })


        # Conver arrays to numpy arrays and convert to float32 type
        image_data = np.array(image_data, dtype="float32")
        image_labels = np.array(image_labels)
        path_list = np.array(path_list)

        # Debug
        print_info(
            "Loaded " + str(len(image_data)) + " images in " +
            str(len(labels_dict)) + " categories",
            1
        )
        for key, value in num_of_images_per_category.items():
            print_info(
                'Category: {0:13} - {1:4d} images'.format(key, value),
                2
            )

        return (image_data, image_labels, labels_dict, path_list)

    @staticmethod
    def load_normalized_images(root_path, width, height):
        # Debug
        print_info("Loading input dataset...")

        image_data = []
        image_paths = glob.glob(root_path + "*")

        for path in image_paths:
            # Load and rescale images
            try:
                image = cv2.resize(cv2.imread(path), (width, height))
                image = img_to_array(image)
                image_data.append(image)
            except:
                print_error('Invalid image: ' + image_path)
                continue

        image_data = np.array(image_data, dtype="float32")
        image_paths = np.array(image_paths)

        # Debug
        print_info("Loaded " + str(len(image_data)) + " images", 1)

        return (image_data, image_paths)

    def generate_angle_images(image_data, angle_range):
        # Debug
        print_info("Generating angle images...")

        angle_images = []
        image_labels = []
        labels_dict = dict(
            (i, angle) for i, angle in enumerate(angle_range)
        )

        for image in image_data:
            for angle in angle_range:
                rotated = rotate(image / 255.0, angle)
                angle_images.append(rotated) # * 255.0)
                image_labels.append(angle)

                cv2.imshow("Rotated (Problematic)", rotated)
                cv2.waitKey(0)

        angle_images = np.array(angle_images, dtype="float32")
        image_labels = np.array(image_labels)

        # Debug
        print_info(
            "Generated " + str(len(angle_images)) + " images" +
            " with angles: " + ', '.join(str(angle) for angle in angle_range),
            1
        )

        return (angle_images, image_labels, labels_dict)

    @staticmethod
    def split_data(training_data, labels, paths, split_size=0.20):
        if (not 0 <= split_size <= 0.5):
            print_error("Invalid split size: " + str(split_size))

        # Debug
        print_info("Spliting input data...")

        train_x, val_x, train_y, val_y, train_p, val_p = train_test_split(
            training_data,
            labels,
            paths,
            test_size=split_size,
            random_state=42
        )
        val_x, test_x, val_y, test_y, val_p, test_p = train_test_split(
            val_x,
            val_y,
            val_p,
            test_size=0.5,
            random_state=42
        )

        # Debug
        print_info(
            '{0:20} {1:4d} ({2:2d}%) images'.format(
                "Training dataset:",
                len(train_x),
                int((1 - split_size) * 100)
            ),
            1
        )
        print_info(
            '{0:20} {1:4d} ({2:2d}%) images'.format(
                "Validation dataset:",
                len(val_x),
                int((split_size * 0.5) * 100)
            ),
            1
        )
        print_info(
            '{0:20} {1:4d} ({2:2d}%) images'.format(
                "Test dataset:",
                len(test_x),
                int((split_size * 0.5) * 100)
            ),
            1
        )

        return (
            train_x, to_categorical(train_y, num_classes=2), train_p,
            val_x, to_categorical(val_y, num_classes=2), val_p,
            test_x, test_y, test_p
        )

    @staticmethod
    def load_and_preprocess_image(image_path, width, height, preproc):
        # Debug
        print_info("Loading image from: " + image_path)

        # Preprocess image
        image = cv2.resize(
            cv2.imread(image_path),
            (width, height)
        )
        image = img_to_array(image.astype("float"))

        if (preproc):
            image = preproc.apply(image)

        return image

    @staticmethod
    def load_model_data(model_path, from_json=False):
        '''Load model data as width, height, depth, labels_dict
        Model can be loaded from .h5 file as trained model or
        from json file as model architecture - model is not trained

        model_path - string, path to model folder
        from_json - boolean, if true untrained model architecture is loaded
        '''
        # Debug
        print_info('Loading model from: ' + model_path)

        # Read model settings file
        json_file = open(model_path + MODEL_SETTINGS_FILE)
        model_data = json.load(json_file)

        # Preprocessor class
        preproc = Preprocessor()
        for func in model_data['preprocessing']['func_list']:
            preproc.add_func(Preprocessing().__getattribute__(func))

        # Datagen class
        preproc.set_datagen(
            default=False,
            **dict(model_data['preprocessing']['datagen_args'])
        )

        # Load (un)trained model
        if (not from_json):
            loaded_model = load_model(model_path + MODEL_BINARY_FILE)
        else:
            # Debug
            print_warning(
                'Loading only model architecture(untrained model)',
                1
            )

            # TODO
            # error dajaky
            with open(model_path + MODEL_ARCHITECTURE_FILE) as jsonfile:
                #loaded_model = model_from_json(str(json.load(jsonfile)))
                pass

        # Model class
        model_class = Model(
            (
                1,
                model_data['width'],
                model_data['height'],
                model_data['depth']
            ),
            model_data['labels'],
            model_folder=model_path,
            model_name=model_data['model_name'],
            model=loaded_model
        )

        model_class.batch_size = model_data['batch_size']
        model_class.epochs = model_data['epochs']
        model_class.set_optimizer(model_data['optimizer'])

        return (
            model_class,
            preproc,
        )
