from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.models import Sequential
from keras import backend as K
from abc import ABCMeta, abstractmethod
from datetime import datetime
from preprocessing import Preprocessor
from printer import print_info, print_warning

import json
import os


class Model():

    __metaclass__ = ABCMeta

    def __init__(self, data_shape, labels_dict,
                 model_folder='', model_name=None, model=None):
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

        self.optimizer = None

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

    def get_model(self):
        return self.model
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_model_folder(self, model_folder_path):
        self.model_folder = model_folder_path

    def __compile(self, loss='binary_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy']):
        self.model.compile(
            loss=loss,
            optimizer=self.optimizer if (self.optimizer) else optimizer,
            metrics=metrics
        )

    def train(self, train_x, train_y, val_x, val_y,
              datagen, epochs, batch_size):
        print_info('\nTraining model...')
        self.__compile()

        return self.model.fit_generator(
            datagen.flow(train_x, train_y, batch_size=batch_size),
            steps_per_epoch=len(train_x) // batch_size,
            epochs=epochs,
            validation_data=(val_x, val_y),
            #validation_steps=len(val_x) // batch_size
        )

    def __get_prediction(self, image):
        switched_labels = dict((y,x) for x,y in self.labels_dict.items())
        result_dict = {}
        for i, value in enumerate(self.model.predict(image)[0]):
            result_dict.update({
                switched_labels[i]: round(float(value), 6)
            })
        return result_dict

    def evaluate(self, test_x, test_y, test_p):
        # Debug
        print_info('Model evaluation...')

        if (len(test_x.shape) == 3):    # Image evaluation
            self.__image_evaluation(test_x)
        else:                           # Dataset evaluation
            self.__dataset_evaluation(test_x, test_y, test_p)

    def __image_evaluation(self, input_image):
        input_image = input_image.reshape((1,) + input_image.shape)
        print_info(self.__get_prediction(input_image), 1)

    def __dataset_evaluation(self, test_x, test_y, test_p):
        # Check input arguments
        if (len(test_x) != len(test_y) != len(test_p)):
            print_warning(
                'Invalid length of input parameters:\n' +
                '\ttest_x (' + str(len(test_x)) + '), ' +
                'test_y (' + str(len(test_y)) + '), ' +
                'test_p (' + str(len(test_p)) + ')', 2
            )

        testing_score = dict(
            (key, {'correct': [], 'wrong': [], 'score': -1})
                for key in self.labels_dict
        )
        switched_labels = dict((y,x) for x,y in self.labels_dict.items())

        for image, label_idx, path in zip(test_x, test_y, test_p):
            # Change shape to (1, x, y, depth)
            image = image.reshape((1,) + image.shape)
            # Image prediction - {'weapon-type': change, ...}
            result_all = self.__get_prediction(image)
            max_key = max(result_all, key=result_all.get)

            # Create dict for summary json file
            output_dict = {'path': path, 'result': result_all}

            if (self.labels_dict[max_key] == label_idx):    # Correct prediction
                testing_score[max_key]['correct'].append(output_dict)
            else:
                testing_score[switched_labels[label_idx]]['wrong'].append(output_dict)

        Model.__save_evaluation_results(self.model_folder, testing_score)

    @staticmethod
    def __save_evaluation_results(model_folder_path, testing_score):
        total_img = 0
        correct_img = 0

        # Stdout summary & calculate final score
        summary = []
        for key, value_list in testing_score.items():
            path_sum = len(value_list['correct']) + len(value_list['wrong'])
            score = round(len(value_list['correct']) / path_sum * 100)

            testing_score[key]['score'] = score
            total_img += path_sum
            correct_img += len(value_list['correct'])

            summary.append(
                key + '\t' + str(score) + '%' +
                '\t(' + str(len(value_list['correct'])) + "/" + str(path_sum) + ')'
            )

        # Print short summary to stdout
        print_info(
            'Accuracy: ' + str(round(correct_img / total_img * 100, 2)) + '%',
            1
        )
        for message in summary:
            print_info(message, 2)

        # Save summary to files
        if (model_folder_path):
            now_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

            logs_folder = model_folder_path + "logs/"
            if (not os.path.exists(logs_folder)):
                os.mkdir(logs_folder)

            logs_folder += now_str + "/"
            if (not os.path.exists(logs_folder)):
                os.mkdir(logs_folder)

            # Debug
            print_info('Testing details in folder: ' + logs_folder, 1)

            summary_file = open(logs_folder + 'summary.json', 'w')
            summary_file.write(
                json.dumps(
                    testing_score,
                    sort_keys=False,
                    indent=4,
                    separators=(',', ':')
                )
            )
            summary_file.close()

            # Concatenate all paths
            path_list = []
            for key, value_list in testing_score.items():
                path_list += [res_dict['path'] for res_dict in value_list['correct']]
                path_list += [res_dict['path'] for res_dict in value_list['wrong']]

            # Save paths to testing data
            testing_data_file = open(logs_folder + 'testing_data.txt', 'w')
            testing_data_file.write('\n'.join(path_list))
        else:
            # Debug
            print_warning(
                'No model folder, evaluation details were not saved'
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
        # Initialize the model
        model = Sequential()

        # first set of CONV => RELU => POOL layers
        model.add(
            Conv2D(
                20,
                (5, 5),
                padding='same',
                input_shape=self.input_shape
        ))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # second set of CONV => RELU => POOL layers
        model.add(
            Conv2D(
                50,
                (5, 5),
                padding='same'
            )
        )
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
        # Input shape 64x64x1
        model = Sequential()

        model.add(Conv2D(
            16, (2, 2), strides=(1, 1), padding='same',
            input_shape=self.input_shape, #activation='relu'
            )
        )   # Output 64x64x16
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))   # Output 32x32x16

        model.add(Conv2D(32, (2, 2), strides=(1, 1), padding='same'))   # Output 32x32x32
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))   # Output 16x16x32

        model.add(Conv2D(64, (2, 2), strides=(1, 1), padding='same'))   # Output 16x16x64
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))   # Output 8x8x64

        model.add(Conv2D(128, (2, 2), strides=(1, 1), padding='same'))   # Output 8x8x128
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))   # Output 4x4x128

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))

        #model.add(Dropout(0.5))
        model.add(Dense(self.num_of_classes))
        model.add(Activation('softmax'))

        return model

