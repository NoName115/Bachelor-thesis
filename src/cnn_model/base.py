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

def translate_prediction(prediction, labels_dict, get_max=False):
    swiched_labels = dict((y,x) for x,y in labels_dict.items())
    result_dict = {}
    for i, value in enumerate(prediction):
        result_dict.update({
            swiched_labels[i]: value
        })
    if (not get_max):
        return result_dict
    else:
        max_key = max(result_dict, key=result_dict.get)
        return [max_key, result_dict[max_key]]
