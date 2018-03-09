from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical, print_summary
from keras.models import load_model, model_from_json
from sklearn.model_selection import train_test_split
from imutils import paths
from printer import print_info, print_warning, print_error
from preprocessing import Preprocessor, Preprocessing
from models import Model
from datetime import datetime
from hashlib import sha1

import numpy as np
import random
import json
import glob
import cv2
import os


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
    def save_model(save_to_folder, model, preprocesor, with_datetime=False):
        model_name = model.get_name()

        # Model folder path
        model_folder_path = save_to_folder + model_name
        if (with_datetime):
            model_folder_path += '_' + datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        if (not os.path.exists(model_folder_path)):
            os.mkdir(model_folder_path)

        # Debug info
        print_info("Saving model to: " + model_folder_path)

        model_file = model_folder_path + '/' + model_name

        # Save model info
        json_file = open(model_file + '.json', 'w')
        json_file.write(
            json.dumps(
                {
                    'model_name': model_name,
                    'width': model.width,
                    'height': model.height,
                    'labels': model.labels_dict,
                    'depth': model.depth,
                    'preprocessing': preprocesor.get_json(),
                },
                sort_keys=False,
                indent=4,
                separators=(',', ':')
            )
        )
        json_file.close()

        # Save model as h5
        model.model.save(model_file + '.h5')

        # Save model summary info
        summary_file = open(model_file + '.summary', 'w')
        print_summary(
            model.model,
            line_length=79,
            positions=None,
            print_fn=lambda in_str: summary_file.write(in_str + '\n')
        )
        summary_file.close()

        # Save model architecture as json
        arch_file = open(model_folder_path + '/architecture.json', 'w')
        arch_file.write(
            json.dumps(
                json.loads(model.model.to_json()),
                sort_keys=False,
                indent=4,
                separators=(',', ':')
            )
        )
        arch_file.close()


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
    def load_scaled_data_with_labels(
        root_path, width, height, correct_dataset_size=True
    ):
        image_data = []
        image_labels = []
        labels_dict = {}
        labels_counter = 0
        num_of_images_per_category = {}

        # DEBUG
        print_info("Creating category dictionary...")

        # Create labels_dict
        for category in os.listdir(root_path):
            labels_dict.update({
                category: labels_counter
            })
            labels_counter += 1

        # DEBUG
        print_info("Category dictionary: " + str(labels_dict), 1)
        print_info("Loading input dataset...")

        # Correct dataset size
        category_counter = []
        max_images = None
        if (correct_dataset_size):
            for cat_path in glob.glob(root_path + '/*'):
                category_counter.append(
                    len(os.listdir(cat_path))
                )
            max_images = min(category_counter)

        # Image hash table
        image_hash_table = {}

        # Grab the image paths and randomly shuffle them
        image_paths = list(paths.list_images(root_path))
        #random.seed(42)
        #random.shuffle(image_paths)

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
                continue

            # Load and rescale images
            image = cv2.resize(cv2.imread(path), (width, height))
            image = img_to_array(image)
            image_data.append(image)

            # Labels operations
            image_labels.append(labels_dict[label])
            num_of_images_per_category[label] += 1

            # Add image to image hash table
            image_hash = sha1(image).hexdigest()

            if (image_hash in image_hash_table):
                print_warning(
                    'Duplicite image: ' + path + " & " + image_hash_table[image_hash],
                    2
                )
                continue

            image_hash_table.update({
                image_hash: path
            })


        # Conver arrays to numpy arrays and convert to float32 type
        image_data = np.array(image_data, dtype="float32")
        image_labels = np.array(image_labels)

        # Debug
        print_info(
            "Loaded " + str(len(image_data)) + " images in " +
            str(len(labels_dict)) + " categories",
            1
        )
        for key, value in num_of_images_per_category.items():
            print_info(
                "Category: " + key + " - " + str(value) + " images",
                2
            )

        # Check number of loaded images with image hash table
        if (len(image_data) != len(image_hash_table)):
            print_warning(
                "Different length of image_data (" + str(len(image_data)) + ") & " +
                "image_hash_table (" + str(len(image_hash_table)) + ")"
            )

        return (image_data, image_labels, labels_dict, image_hash_table)

    @staticmethod
    def split_data(training_data, labels, split_size=0.20):
        if (not 0 <= split_size <= 0.5):
            print_error("Invalid split size: " + str(split_size))

        # Debug
        print_info("Spliting input data...")

        train_x, val_x, train_y, val_y = train_test_split(
            training_data,
            labels,
            test_size=split_size,
            random_state=42
        )
        val_x, test_x, val_y, test_y = train_test_split(
            val_x,
            val_y,
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
            train_x,
            to_categorical(train_y, num_classes=2),
            val_x,
            to_categorical(val_y, num_classes=2),
            test_x,
            test_y
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

        # Get model name
        model_name = model_path.split(os.path.sep)[-2]

        # Read data from json file
        json_file = open(model_path + model_name + '.json')
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
            loaded_model = load_model(model_path + model_name + '.h5')
        else:
            # Debug
            print_warning(
                'Loading only model architecture(untrained model) from json',
                1
            )

            # TODO
            # error dajaky
            with open(model_path + 'architecture.json') as jsonfile:
                #loaded_model = model_from_json(str(json.load(jsonfile)))
                pass

        # Model class
        model_class = Model(
            model_data['width'],
            model_data['height'],
            model_data['labels'],
            depth=model_data['depth'],
            model_name=model_name,
            model=loaded_model
        )

        return (
            model_class,
            preproc,
        )
