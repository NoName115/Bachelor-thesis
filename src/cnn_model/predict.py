from keras.models import load_model
from base import parse_arguments_prediction
from loader import DataLoader
from printer import print_info, print_error


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
        preproc
    )
    model_class.evaluate(
        image, None, None
    )

# Predict more images
if (args['dataset']):
    print_info('Dataset prediction...')
    image_data, image_labels, _, path_list = DataLoader.load_scaled_data_with_labels(
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

# Predict images from file
if (args['file']):
    print_info('File prediction...')
    image_data, image_labels, _, path_list = DataLoader.load_scaled_data_from_file(
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
