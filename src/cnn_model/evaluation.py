from printer import print_info, print_error
from preprocessing import Preprocessing
from base import calculate_diff_angle, get_prediction, Algorithm
from datetime import datetime
from sklearn import metrics

import json
import os
import cv2


def evaluate_model(model_class, test_x, test_y, test_p, threshold=5):
    if (model_class.algorithm == Algorithm.SVM):
        __evaluate_SVM(model_class, test_x, test_y, test_p)
    elif (model_class.algorithm == Algorithm.KMEANS):
        __evalute_KMeans(model_class, test_x, test_y, test_p)
    elif (model_class.algorithm == Algorithm.MLP):
        __evaluate_MLP(model_class, test_x, test_y, test_p)
    elif (model_class.algorithm == Algorithm.CNN):
        if (model_class.model_type == "class"):
            __evalute_classification(model_class, test_x, test_y, test_p)
        elif (model_class.model_type == "angle"):
            __evaluate_angle(model_class, test_x, test_p, threshold)
        else:
            print_error("Invalid model type")
    else:
        print_error("Unknown learning algorithm")

def __evaluate_SVM(model_class, test_x, test_y, test_p):
    print_info("Model SVM evaluation...")

    testing_score = dict(
        (key, {'correct': [], 'wrong': []})
            for key in model_class.labels_dict
    )
    switched_labels = dict((y,x) for x,y in model_class.labels_dict.items())

    prediction = list(model_class.model.predict(test_x))
    expectation = list(test_y)

    for predict, expect, path in zip(prediction, expectation, test_p):
        # Create dict for summary json file
        output_dict = {'path': path, 'result': switched_labels[predict]}

        if (predict == expect): # Correct prediction
            testing_score[switched_labels[predict]]['correct'].append(
                output_dict
            )
        else:                   # Wrong prediciton
            testing_score[switched_labels[expect]]['wrong'].append(
                output_dict
            )

    __save_evalution_class_results(model_class.model_folder, testing_score)

def __evalute_KMeans(model_class, test_x, test_y, test_p):
    print_info("Model KMeans evaluation...")

    testing_score = dict(
        (key, {'correct': [], 'wrong': []})
            for key in model_class.labels_dict
    )
    switched_labels = dict((y,x) for x,y in model_class.labels_dict.items())

    prediction = list(model_class.model.predict(test_x))
    expectation = list(test_y)

    for predict, expect, path in zip(prediction, expectation, test_p):
        # Create dict for summary json file
        output_dict = {'path': path, 'result': switched_labels[predict]}

        if (predict == expect): # Correct prediction
            testing_score[switched_labels[predict]]['correct'].append(
                output_dict
            )
        else:                   # Wrong prediciton
            testing_score[switched_labels[expect]]['wrong'].append(
                output_dict
            )

    __save_evalution_class_results(model_class.model_folder, testing_score)

def __evaluate_MLP(model_class, test_x, test_y, test_p):
    print_info("Model MLPerceptron evaluation...")

    testing_score = dict(
        (key, {'correct': [], 'wrong': []})
            for key in model_class.labels_dict
    )
    switched_labels = dict((y,x) for x,y in model_class.labels_dict.items())

    prediction = list(model_class.model.predict(test_x))
    expectation = list(test_y)

    for predict, expect, path in zip(prediction, expectation, test_p):
        # Create dict for summary json file
        output_dict = {'path': path, 'result': switched_labels[predict]}

        if (predict == expect): # Correct prediction
            testing_score[switched_labels[predict]]['correct'].append(
                output_dict
            )
        else:                   # Wrong prediciton
            testing_score[switched_labels[expect]]['wrong'].append(
                output_dict
            )

    __save_evalution_class_results(model_class.model_folder, testing_score)

def __evalute_classification(model_class, test_x, test_y, test_p):
    # Debug
    print_info('Model classification evaluation...')

    if (len(test_x.shape) == 3):    # Image evaluation
        __evalute_classification_image(model_class, test_x)
    else:                           # Dataset evaluation
        __evalute_classification_dataset(model_class, test_x, test_y, test_p)

def __evalute_classification_image(model_class, input_image):
    input_image = input_image.reshape((1,) + input_image.shape)
    print_info(get_prediction(model_class, input_image), 1)

    cv2.imshow('Classification image', input_image[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def __evalute_classification_dataset(model_class, test_x, test_y, test_p):
    testing_score = dict(
        (key, {'correct': [], 'wrong': []})
            for key in model_class.labels_dict
    )
    switched_labels = dict((y,x) for x,y in model_class.labels_dict.items())

    for image, label_idx, path in zip(test_x, test_y, test_p):
        # Change shape to (1, x, y, depth)
        image = image.reshape((1,) + image.shape)
        # Image prediction - {'weapon-type': change, ...}
        result_all = get_prediction(model_class, image)
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

def __evaluate_angle(model_class, test_x, test_p, threshold):
    # Debug
    print_info('Model angle evaluation...')
    print_info('Threshold: ' + str(threshold), 1)

    if (len(test_x.shape) == 3):    # Image evaluation
        __evaluate_angle_image(model_class, test_x, threshold)
    else:                           # Dataset evaluation
        __evaluate_angle_dataset(model_class, test_x, test_p, threshold)

def __evaluate_angle_image(model_class, image, threshold):
    # Rotate image & change shape to (1, x, y, depth)
    rotated, angle, _ = Preprocessing.rotate_and_crop_image(
        image, model_class.labels_dict
    )
    rotated = rotated.reshape((1,) + rotated.shape)

    # Image prediction - {angle: change, ...}
    result_all = get_prediction(model_class, rotated)
    max_key = max(result_all, key=result_all.get)

    # Calculate diff. angle
    pred_angle = int(max_key)
    diff_angle = calculate_diff_angle(angle, pred_angle)

    output_dict = {
        'correct': angle,
        'predict': pred_angle,
        'diff': diff_angle
    }
    print_info(output_dict, 1)
    print_info("Threshold: " + str(threshold), 1)

    cv2.imshow('Angle image', rotated[0])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def __evaluate_angle_dataset(model_class, test_x, test_p, threshold):
    testing_score = {'correct': [], 'wrong': []}
    angle_range = list(model_class.labels_dict.keys())

    for image, path in zip(test_x, test_p):
        # Rotate image & change shape to (1, x, y, depth)
        rotated, angle, _ = Preprocessing.rotate_and_crop_image(
            image, model_class.labels_dict
        )
        rotated = rotated.reshape((1,) + rotated.shape)

        # Image prediction - {angle: change, ...}
        result_all = get_prediction(model_class, rotated)
        max_key = max(result_all, key=result_all.get)

        # Calculate diff. angle
        pred_angle = int(max_key)
        diff_angle = calculate_diff_angle(angle, pred_angle)

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
    # Check input path
    if (not model_folder_path):
        print_warning(
            'No model folder, evaluation details were not saved'
        )
        return

    # Save summary to files
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
    testing_data_file.write('\n'.join(set(path_list)))
