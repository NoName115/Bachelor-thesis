from datetime import datetime
from printer import print_info, print_warning
from loader import DataLoader

import argparse
import json
import os


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

def test_training(test_x, test_y, test_p, model_folder_path, preprocessed=False):
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

    # Load model & preprocess data
    model_class, preproc = DataLoader.load_model_data(model_folder_path)
    if (not preprocessed):
        test_x = preproc.apply(test_x)

    labels_dict = model_class.labels_dict
    testing_score = dict(
        (key, {'correct': [], 'wrong': [], 'score': -1}) for key in labels_dict
    )
    switched_labels = __switch_dict(labels_dict)

    for image, label_idx, path in zip(test_x, test_y, test_p):
        # Change shape to (1, x, y, depth)
        image = image.reshape((1,) + image.shape)
        result = translate_prediction(
            model_class.model.predict(image)[0],
            labels_dict,
            get_max=True
        )   # result - ['weapon-type', change]

        if (labels_dict[result[0]] == label_idx):    # Correct prediction
            testing_score[result[0]]['correct'].append(path)
        else:
            testing_score[switched_labels[label_idx]]['wrong'].append(path)

    # Save testing summary
    __save_testing_results(model_folder_path, testing_score)

def __save_testing_results(model_folder_path, testing_score):
    now_str = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    logs_folder = model_folder_path + "logs/"
    if (not os.path.exists(logs_folder)):
        os.mkdir(logs_folder)

    logs_folder += now_str + "/"
    if (not os.path.exists(logs_folder)):
        os.mkdir(logs_folder)

    # Debug
    print_info('Testing details in folder: ' + logs_folder, 1)

    # Stdout summary & calculate final score
    summary = []
    for key, value_list in testing_score.items():
        path_sum = len(value_list['correct']) + len(value_list['wrong'])
        score = round(len(value_list['correct']) / path_sum * 100)
        testing_score[key]['score'] = score
        summary.append(
            key + '\t' + str(score) + '%' +
            '\t(' + str(len(value_list['correct'])) + "/" + str(path_sum) + ')'
        )

    # Print short summary to stdout
    for message in summary:
        print_info(message, 1)

    # Save summary to file
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
        path_list += value_list['correct'] + value_list['wrong']

    # Save paths to testing data
    testing_data_file = open(logs_folder + 'testing_data.txt', 'w')
    testing_data_file.write('\n'.join(path_list))
