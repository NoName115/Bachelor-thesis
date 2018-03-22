from printer import print_info, print_error
from preprocessing import Preprocessing
from imutils import rotate_bound
from datetime import datetime

import numpy as np
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
        "--type",
        required=True,
        help="train classificator(class) or angle detector(angle)",
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
    ap.add_argument(
        "--file",
        required=False,
        help="path to file with paths to images"
    )
    return vars(ap.parse_args())

def evaluate_model(model_class, test_x, test_y, test_p):
    if (model_class.model_type == "class"):
        # TODO for image
        __evalute_classification(model_class, test_x, test_y, test_p)
    elif (model_class.model_type == "angle"):
        # TODO for image
        __evaluate_angle(model_class, test_x, test_p)
    else:
        print_error("Invalid model type")

def __get_prediction(model_class, image):
    switched_labels = dict((y,x) for x,y in model_class.labels_dict.items())
    result_dict = {}
    for i, value in enumerate(model_class.model.predict(image)[0]):
        result_dict.update({
            switched_labels[i]: round(float(value), 6)
        })
    return result_dict

'''
def __evalute_classification(model_class, test_x, test_y, test_p):
    # Debug
    print_info('Model classification evaluation...')

    if (len(test_x.shape) == 3):    # Image evaluation
        __image_evaluation(test_x)
    else:                           # Dataset evaluation
        __dataset_evaluation(test_x, test_y, test_p)


def __image_evaluation(input_image):
    input_image = input_image.reshape((1,) + input_image.shape)
    print_info(__get_prediction(input_image), 1)
'''

def __evalute_classification(model_class, test_x, test_y, test_p):
    # Debug
    print_info('Model classification evaluation...')

    testing_score = dict(
        (key, {'correct': [], 'wrong': []})
            for key in model_class.labels_dict
    )
    switched_labels = dict((y,x) for x,y in model_class.labels_dict.items())

    for image, label_idx, path in zip(test_x, test_y, test_p):
        # Change shape to (1, x, y, depth)
        image = image.reshape((1,) + image.shape)
        # Image prediction - {'weapon-type': change, ...}
        result_all = __get_prediction(model_class, image)
        max_key = max(result_all, key=result_all.get)

        # Create dict for summary json file
        output_dict = {'path': path, 'result': result_all}

        if (model_class.labels_dict[max_key] == label_idx):    # Correct prediction
            testing_score[max_key]['correct'].append(output_dict)
        else:
            testing_score[switched_labels[label_idx]]['wrong'].append(output_dict)

    __save_evalution_class_results(model_class.model_folder, testing_score)

def __save_evalution_class_results(model_folder_path, testing_score):
    total_img = 0
    correct_img = 0

    # Stdout summary & calculate final score
    summary = []
    for key, value_list in testing_score.items():
        path_sum = len(value_list['correct']) + len(value_list['wrong'])
        score = round(len(value_list['correct']) / path_sum * 100)

        total_img += path_sum
        correct_img += len(value_list['correct'])

        summary.append(
            str(key) + '\t' + str(score) + '%' +
            '\t(' + str(len(value_list['correct'])) + "/" + str(path_sum) + ')'
        )

    # Print short summary to stdout
    print_info(
        'Accuracy: ' + str(round(correct_img / total_img * 100, 2)) + '%',
        1
    )
    for message in summary:
        print_info(message, 2)

    # Concatenate all paths
    path_list = []
    for key, value_list in testing_score.items():
        path_list += [res_dict['path'] for res_dict in value_list['correct']]
        path_list += [res_dict['path'] for res_dict in value_list['wrong']]

    __save_log_files(model_folder_path, testing_score, path_list)

def __evaluate_angle(model_class, test_x, test_p, threshold=18):
    # Debug
    print_info('Model angle evaluation...')

    testing_score = {'correct': [], 'wrong': []}
    angle_range = list(model_class.labels_dict.keys())

    for image, path in zip(test_x, test_p):
        # Get random angle
        angle = int(angle_range[np.random.randint(0, len(angle_range))])
        # Rotate image & change shape to (1, x, y, depth)
        rotated = Preprocessing.rotate_and_crop_image(image, angle)
        rotated = rotated.reshape((1,) + rotated.shape)

        # Image prediction - {angle: change, ...}
        result_all = __get_prediction(model_class, rotated)
        max_key = max(result_all, key=result_all.get)

        # Calculate diff. angle
        pred_angle = int(max_key)
        diff_angle = abs(angle - pred_angle)
        if (diff_angle > 180):
            diff_angle = abs(diff_angle - 360)

        output_dict = {
            'path': path,
            'correct': angle,
            'predict': pred_angle,
            'diff': diff_angle
        }

        if (diff_angle <= threshold):   # Correct prediction
            testing_score['correct'].append(output_dict)
        else:                           # Wrong prediction
            testing_score['wrong'].append(output_dict)

    __save_evalution_angle_results(model_class.model_folder, testing_score)

def __save_evalution_angle_results(model_folder_path, testing_score):
    # Print short summary to stdout
    path_sum = len(testing_score['correct']) + len(testing_score['wrong'])
    print_info(
        "Accuracy: " + str(
            round(len(testing_score['correct']) / path_sum * 100, 2)
        ) + '%',
        1
    )
    print_info("Correct: " + str(len(testing_score['correct'])), 2)
    print_info("Wrong: " + str(len(testing_score['wrong'])), 2)

    # Concatenate all paths
    path_list = []
    path_list += [record['path'] for record in testing_score['correct']]
    path_list += [record['path'] for record in testing_score['wrong']]

    __save_log_files(model_folder_path, testing_score, path_list)

def __save_log_files(model_folder_path, testing_score, path_list):
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

        # Save paths to testing data
        testing_data_file = open(logs_folder + 'testing_data.txt', 'w')
        testing_data_file.write('\n'.join(path_list))
    else:
        # Debug
        print_warning(
            'No model folder, evaluation details were not saved'
        )
