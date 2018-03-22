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

def evaluate_angle(model_class, test_x):
    from preprocessing import Preprocessing
    from imutils import rotate_bound
    from printer import print_info
    import numpy as np
    import cv2

    print_info("Angle evaluation...")

    angle_range = list(model_class.labels_dict.keys())
    for image in test_x:
        print("--------- IMAGE ----------")

        angle = angle_range[np.random.randint(0, len(angle_range))]
        rotated = Preprocessing.crop_rotated_image(
            rotate_bound(image, angle),
            angle,
            image.shape[0],
            image.shape[1]
        )
        rotated = rotated.reshape((1,) + rotated.shape)

        #print(model_class.model.predict(rotated)[0])
        '''
        prediction = model_class.model.predict(rotated)[0]
        max_value = max(prediction)
        max_indx = list(prediction).index(max_value)
        print("M: " + str(max_value))
        print("I: " + str(max_indx))
        print("IV: " + str(prediction[max_indx]))
        print("O: " + str(angle))
        '''

        preds = model_class.model.predict(rotated)
        preds[preds>=0.5] = 1
        preds[preds<0.5] = 0

        print(preds)

        rotated = rotated.reshape(
            (rotated.shape[1], rotated.shape[2], rotated.shape[3])
        )

        cv2.imshow('Rotated', rotated)# / 255.0)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
