from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical, print_summary
from keras.models import load_model, model_from_json
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from imutils import paths
from datetime import datetime
from hashlib import sha1
from printer import print_info, print_warning, print_error, print_blank
from preprocessing import Preprocessor, Preprocessing
from models import Model
from base import angle_error, Algorithm

import matplotlib.pyplot as plt
import numpy as np
import json
import glob
import os
import re

a = {
    'a': 2,
    'b': 3
}

MODEL_SETTINGS_FILE = 'settings.json'
MODEL_BINARY_FILE = 'model.h5'
MODEL_ARCHITECTURE_FILE = 'architecture.json'
MODEL_SUMMARY_FILE = 'summary.txt'
MODEL_TRAINING_PLOT = 'training_history.png'

class DataSaver():

    @staticmethod
    def save_image(image, path, image_name, rescale=True):
        '''Save input array as image, only for PNG image extention
        '''
        import cv2
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
             save_to_dir=folder_path, save_prefix='weapon', save_format='jpeg'):
            # Debug
            if (i % print_mod == 0):
                print_info("Saving loop " + str(i), 1)
            i += 1
            if i > loops:
                break

    @staticmethod
    def save_model(save_to_folder, model_class, preprocesor,
                   training_history=None, with_datetime=False):
        model_name = model_class.model_name

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
                    'name': model_name,
                    'type': model_class.model_type,
                    'algorithm': model_class.algorithm,
                    'rotation_type': model_class.rotation_type,
                    'width': model_class.width,
                    'height': model_class.height,
                    'labels': model_class.labels_dict,
                    'depth': model_class.depth,
                    'batch_size': model_class.batch_size,
                    'epochs': model_class.epochs,
                    'preprocessing': preprocesor.get_json(),
                },
                sort_keys=False,
                indent=4,
                separators=(',', ':')
            )
        )
        json_file.close()

        # Save binary model
        if (model_class.algorithm == Algorithm.CNN):
            model_class.model.save(
                model_folder_path + MODEL_BINARY_FILE
            )
            DataSaver.__save_addional_info_CNN(
                model_class,
                model_folder_path,
                training_history
            )
        elif (model_class.algorithm == Algorithm.SVM or
              model_class.algorithm == Algorithm.KMEANS or
              model_class.algorithm == Algorithm.MLP):
            joblib.dump(
                model_class.model,
                model_folder_path + MODEL_BINARY_FILE
            )
        else:
            print_error("Unkown learning algorithm")

        # Return path where model was saved
        return model_folder_path

    @staticmethod
    def __save_addional_info_CNN(model_class, model_folder_path,
                                 training_history):
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

        # Save training history to PNG
        plt.style.use("ggplot")
        plt.figure()
        N = model_class.epochs

        history_keys = list(training_history.history.keys())

        plt.plot(
            np.arange(0, N),
            training_history.history["loss"],
            label="train_loss"
        )
        plt.plot(
            np.arange(0, N),
            training_history.history["val_loss"],
            label="val_loss"
        )
        plt.plot(
            np.arange(0, N),
            training_history.history[history_keys[3]],
            label=history_keys[3]
        )
        plt.plot(
            np.arange(0, N),
            training_history.history[history_keys[1]],
            label=history_keys[1]
        )

        plt.title("Training history")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="lower left")
        plt.savefig(model_folder_path + MODEL_TRAINING_PLOT)


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
    def load_images_from_file(file_path, width, height,
                              labels_dict, angle_images):
        path_list = []

        with open(file_path, 'r') as image_file:
            path_list = [line.rstrip('\n') for line in image_file]

        return DataLoader.__load_images_by_path(
            path_list, width, height, labels_dict=labels_dict,
            angle_images=angle_images
        )

    @staticmethod
    def load_images_from_folder(folder_path, width, height, labels_dict={},
                                create_labels=True, correct_dataset_size=True,
                                angle_images=False):
        # Correction cant be done with no labels
        if (correct_dataset_size and not create_labels):
            correct_dataset_size = False

        # Create labels_dict
        labels_dict_out = None
        if (create_labels or labels_dict) and (not angle_images):
            print_info("Creating category dictionary...")                
            labels_dict_out = dict(
                (category, i) for i, category in enumerate(
                    os.listdir(folder_path)
                )
            ) if (not labels_dict) else labels_dict
            print_info("Categories: " + str(labels_dict_out), 1)

        if (angle_images):
            labels_dict_out = labels_dict

        # Correct dataset size
        max_images = 0
        if (create_labels and correct_dataset_size):
            category_counter = [
                len(os.listdir(cat_path))
                    for cat_path in glob.glob(folder_path + '/*')
            ]
            max_images = min(category_counter)

        # Grab the image paths and randomly shuffle them
        image_paths = list(paths.list_images(folder_path))

        return DataLoader.__load_images_by_path(
            image_paths, width, height,
            labels_dict=labels_dict_out,
            correct_dataset_size=correct_dataset_size,
            max_images=max_images,
            angle_images=angle_images,
        )

    @staticmethod
    def __load_images_by_path(image_paths, width, height, labels_dict=None,
                              correct_dataset_size=False, max_images=0,
                              angle_images=False):
        print_info("Loading input dataset...")

        # Initialize data variables
        image_labels = []
        if (labels_dict and not angle_images):
            num_of_images_per_category = dict(
                (label, 0) for label in labels_dict
            )

        image_data = []
        path_list = []
        image_hash_table = {}

        for path in image_paths:
            # Load image label
            if (labels_dict and not angle_images):
                label = path.split(os.path.sep)[-2]

                # Correct dataset size
                if (correct_dataset_size and
                   num_of_images_per_category[label] >= max_images):
                    continue

            # Load and rescale images
            try:
                image = image.load_img(path, target_size=(width, height))
                image = img_to_array(image)
                image_data.append(image)
            except:
                print_error('Invalid image: ' + path)
                continue

            # Labels operations
            if (labels_dict and not angle_images):
                image_labels.append(labels_dict[label])
                num_of_images_per_category[label] += 1

            # Load label from path
            if (labels_dict and angle_images):
                angle_type, image_name = path.split(os.path.sep)[-2:]
                angle_char = angle_type[0]
                axis = re.search(angle_char + '(\d+\.\d+)', image_name)
                angle = int(float(axis.groups()[0]))
                image_labels.append(
                    Preprocessing.get_correct_angle_label(angle, labels_dict)
                )

            # Add path of loaded image
            path_list.append(path)

            # Image duplice detection
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
        path_list = np.array(path_list)

        if (labels_dict and not angle_images):
            image_labels = np.array(image_labels)

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
        else:
            image_labels = np.array(image_labels)

            print_info(
                "Loaded " + str(len(image_data)) + " images",
                1
            )

            return (image_data, image_labels, path_list)
            #return (image_data, [0] * len(image_data), path_list)

    @staticmethod
    def load_angle_images(folder_path, width, height, angle_range, angle_type=[]):
        labels_dict = dict(
            (angle, i) for i, angle in enumerate(angle_range)
        )

        if ('pitch' in angle_type):
            image_data, image_labels, image_paths = DataLoader.load_images_from_folder(
                folder_path + "pitch/", width, height,
                labels_dict=labels_dict,
                create_labels=False,
                angle_images=True
            )
        if ('roll' in angle_type):
            image_data, image_labels, image_paths = DataLoader.load_images_from_folder(
                folder_path + "roll/", width, height,
                labels_dict=labels_dict,
                create_labels=False,
                angle_images=True
            )
        if ('yaw' in angle_type):
            image_data, image_labels, image_paths = DataLoader.load_images_from_folder(
                folder_path + "yaw/", width, height,
                labels_dict=labels_dict,
                create_labels=False,
                angle_images=True
            )

        return (image_data, image_labels, labels_dict, image_paths)

    @staticmethod
    def split_data(training_data, labels, paths, num_classes,
                   split_size=0.20, use_to_categorical=True,
                   get_test=True):
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
        if (get_test):
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
        if (get_test):
            print_info(
                '{0:20} {1:4d} ({2:2d}%) images'.format(
                    "Test dataset:",
                    len(test_x),
                    int((split_size * 0.5) * 100)
                ),
                1
            )

        if (use_to_categorical):
            train_y = to_categorical(train_y, num_classes=num_classes)
            val_y = to_categorical(val_y, num_classes=num_classes)

        result = (
            train_x, train_y, train_p,
            val_x, val_y, val_p
        )
        if (get_test):
            result += (test_x, test_y, test_p)

        return result

    @staticmethod
    def load_image(image_path, width=None, height=None):
        # Debug
        print_info("Loading image from: " + image_path)

        # Preprocess image
        if (not width or not height):
            image = load_img(image_path)
        else:
            image = load_img(image_path, target_size=(width, height))
        return img_to_array(image)

    @staticmethod
    def load_model_data(model_path):
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

        if (model_data['algorithm'] == Algorithm.CNN):
            loaded_model = load_model(
                model_path + MODEL_BINARY_FILE,
                custom_objects={'angle_error': angle_error}
            )
        elif (model_data['algorithm'] == Algorithm.SVM or
              model_data['algorithm'] == Algorithm.KMEANS or
              model_data['algorithm'] == Algorithm.MLP):
            loaded_model = joblib.load(
                model_path + MODEL_BINARY_FILE
            )
        else:
            print_error('Unknown algorithm')

        # Model class
        model_class = Model(
            (
                1,
                model_data['width'],
                model_data['height'],
                model_data['depth']
            ),
            model_data['labels'],
            model_data['type'],
            model_name=model_data['name'],
            model_folder=model_path
        )

        model_class.model = loaded_model
        model_class.batch_size = model_data['batch_size']
        model_class.epochs = model_data['epochs']
        model_class.algorithm = model_data['algorithm']
        model_class.rotation_type = model_data['rotation_type']

        return (
            model_class,
            preproc,
        )
