from printer import print_error

import keras.backend as K
import argparse


def parse_arguments_training():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        required=True,
        help="path to input data for training",
        )
    ap.add_argument(
        "--model",
        required=True,
        help="path where model will be saved",
    )
    ap.add_argument(
        "--alg",
        required=True,
        help="type of algorithm for image classification",
    )
    ap.add_argument(
        "--ep",
        required=False,
        help="number of epochs"
        )
    ap.add_argument(
        "--bs",
        required=False,
        help="batch size"
        )
    return vars(ap.parse_args())

def parse_arguments_prediction():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--class",
        "--c",
        required=True,
        help="path to model for weapon classification"
    )
    ap.add_argument(
        "--anglex",
        "--x",
        required=True,
        help="path to model prediction X axis of weapon angle"
    )
    ap.add_argument(
        "--angley",
        "--y",
        required=True,
        help="path to model prediction Y axis of weapon angle"
    )
    ap.add_argument(
        "--anglez",
        "--z",
        required=True,
        help="path to model prediction Z axis of weapon angle"
    )
    ap.add_argument(
        "--image",
        "--i",
        required=True,
        help="path to model prediction Z axis of weapon angle"
    )
    return vars(ap.parse_args())

def parse_arguments_evaluation():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        required=True,
        help="path to folder of trained model"
    )
    ap.add_argument(
        "--image",
        required=False,
        help="path to input image"
    )
    ap.add_argument(
        "--dataset",
        required=False,
        help="path to input images"
    )
    ap.add_argument(
        "--file",
        required=False,
        help="path to file with paths to images"
    )
    ap.add_argument(
        "--th",
        required=False,
        help="threshold for angle prediction"
    )
    return vars(ap.parse_args())

def angle_error(true_y, pred_y):
    diff = calculate_diff_angle(K.argmax(true_y), K.argmax(pred_y))
    return K.mean(K.cast(K.abs(diff), K.floatx()))

def calculate_diff_angle(correct_angle, predicted_angle):
    # Calculate diff. angle
    return 180 - abs(abs(correct_angle - predicted_angle) - 180)

def get_prediction(model_class, image):
    if (len(image.shape) == 3):
        image = image.reshape(image.shape + (1,))

    switched_labels = dict((y,x) for x,y in model_class.labels_dict.items())
    result_dict = {}
    for i, value in enumerate(model_class.model.predict(image)[0]):
        result_dict.update({
            switched_labels[i]: round(float(value), 6)
        })
    return result_dict

class Algorithm():

    CNN = 'Convolutial Neural Network'
    CNN_C = CNN + ' class'
    CNN_A = CNN + ' angle'
    MLP = 'Multi Layer Perceptron'
    SVM = 'Super Vector Machine'
    KMEANS = 'K-means Classification'

    translate_dict = {
        'cnnc': CNN_C,
        'cnna': CNN_A,
        'mlp': MLP,
        'svm': SVM,
        'kmeans': KMEANS,
    }

    @classmethod
    def translate(self, in_type):
        try:
            return self.translate_dict[in_type]
        except KeyError:
            print_error('Invalid algorithm type')
