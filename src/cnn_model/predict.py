from keras.models import load_model
from base import parse_arguments_prediction
from loader import DataLoader
from printer import print_info, print_error
from imutils import rotate
import cv2


# Parse and check input arguments
# --model, --image, --dataset
args = parse_arguments_prediction()
if (not args['image'] and not args['dataset'] and not args['file']):
    print_error('No input image or dataset to be tested')

# Load model & model data
model_class, preproc = DataLoader.load_model_data(
    args['model']
)
image_width = model_class.width
image_height = model_class.height

# Predict one image
if (args['image']):
    print_info('Image prediction...')
    image = DataLoader.load_and_preprocess_image(
        args['image'],
        image_width,
        image_height,
        None #preproc
    )

    image_rotated = rotate(image, angle)

    model_class.evaluate(
        image, None, None
    )

# Predict more images
if (args['dataset']):
    print_info('Dataset prediction...')
    '''
    image_data, image_labels, _, path_list = DataLoader.load_images_from_folder(
        args['dataset'],
        image_width,
        image_height,
        labels_dict=model_class.labels_dict,
        correct_dataset_size=False
    )
    model_class.evaluate(
        preproc.apply(image_data),
        image_labels,
        path_list
    )
    '''
    image_data, _ = DataLoader.load_normalized_images(
        args['dataset'],
        image_width,
        image_height
    )
    image_data, image_labels, _ = DataLoader.generate_angle_images(
        image_data,
        None,
        labels_dict=model_class.labels_dict
    )

    print(model_class.labels_dict)

    for image in image_data:
        image = image.reshape((1,) + image.shape)
        print(model_class.model.predict(image)[0])
        image = image.reshape((image.shape[1], image.shape[2], image.shape[3]))
        cv2.imshow("Loaded", image)
        cv2.waitKey(0)

# Predict images from file
if (args['file']):
    print_info('File prediction...')
    image_data, image_labels, _, path_list = DataLoader.load_images_from_file(
        args['file'],
        image_width,
        image_height,
        labels_dict=model_class.labels_dict
    )
    model_class.evaluate(
        preproc.apply(image_data),
        image_labels,
        path_list
    )
