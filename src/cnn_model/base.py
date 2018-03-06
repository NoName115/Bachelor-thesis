from printer import print_info

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
        "--graph",
        required=False,
        help="path to output graph about learning",
        default='learning.png'
        )
    return vars(ap.parse_args())

def parse_arguments_prediction():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m", "--model",
        required=True,
        help="path to trained model model"
    )
    ap.add_argument(
        "-i", "--image",
        required=True,
        help="path to input image"
    )
    return vars(ap.parse_args())

def __switch_dict(input_dict):
    return dict((y,x) for x,y in input_dict.items())

def translate_prediction(prediction, labels_dict, get_max=False):
    switched_labels = __switch_dict(labels_dict)
    result_dict = {}
    for i, value in enumerate(prediction):
        result_dict.update({
            switched_labels[i]: value
        })
    if (not get_max):
        return result_dict
    else:
        max_key = max(result_dict, key=result_dict.get)
        return [max_key, result_dict[max_key]]

def test_training(test_x, test_y, model, labels_dict):
    # Debug
    print_info('Final validation score...')

    # List [OK, SUM] & switched labels
    test_score = dict((key, [0, 0]) for key in labels_dict)
    switched_labels = __switch_dict(labels_dict)

    for image, label in zip(test_x, test_y):
        # Change shape to (1, x, y, depth)
        image = image.reshape((1,) + image.shape)
        result = translate_prediction(
            model.predict(image)[0],
            labels_dict,
            get_max=True
        )   # result - ['weapon-type', change]

        if (labels_dict[result[0]] == label):
            test_score[result[0]][0] += 1

        test_score[switched_labels[label]][1] += 1

    for key, value_list in test_score.items():
        print_info(
            key + '\t' + str(round(value_list[0] / value_list[1] * 100)) + '%' +
            '\t(' + str(value_list[0]) + '/' + str(value_list[1]) + ')',
            1
        )
