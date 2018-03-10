from printer import print_info, print_warning

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

def test_training(test_x, test_y, test_p, model, labels_dict):
    # Debug
    print_info('Final validation score...')

    # Check input arguments
    if (len(test_x) != len(test_y) != len(test_p)):
        print_warning(
            'Invalid length of input parameters:\n' +
            '\ttest_x (' + str(len(test_x)) + '), ' +
            'test_y (' + str(len(test_y)) + '), ' +
            'test_p (' + str(len(test_p)) + ')', 2
        )

    # List [OK, SUM, path_list] & switched labels
    test_score = dict((key, [0, 0, []]) for key in labels_dict)
    switched_labels = __switch_dict(labels_dict)

    for image, label, path in zip(test_x, test_y, test_p):
        # Change shape to (1, x, y, depth)
        image = image.reshape((1,) + image.shape)
        result = translate_prediction(
            model.predict(image)[0],
            labels_dict,
            get_max=True
        )   # result - ['weapon-type', change]

        if (labels_dict[result[0]] == label):   # Correct predcition
            test_score[result[0]][0] += 1
        else:                                   # Wrong prediction
            test_score[switched_labels[label]][2].append(
                path
            )

        test_score[switched_labels[label]][1] += 1

    # Output summary
    for key, value_list in test_score.items():
        print_info(
            key + '\t' + str(round(value_list[0] / value_list[1] * 100)) + '%' +
            '\t(' + str(value_list[0]) + '/' + str(value_list[1]) + ')',
            1
        )
        # Invalid predicted images
        print_info('Invalid images:', 2)
        for image_path in value_list[2]:
            print_info(image_path, 3)
