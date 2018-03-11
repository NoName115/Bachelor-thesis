from keras.models import load_model
from .base import parse_arguments_prediction, translate_prediction, test_training
from .loader import DataLoader
from .printer import print_info, print_error


# Parse and check input arguments
# --model, --image, --dataset
args = parse_arguments_prediction()
if (not args['image'] and not args['dataset']):
    print_error('No input image or dataset to be tested')

# Load model & model data
model_class, preproc = DataLoader.load_model_data(
    args['model']
)
image_width = model_class.width
image_height = model_class.height
labels = model_class.labels_dict
model = model_class.get_model()

# Predict one image
if (args['image']):
    print_info('Image prediction...')
    image = DataLoader.load_and_preprocess_image(
        args['image'],
        model_class.width,
        model_class.height,
        preproc
    )

    result = model.predict(image)[0]
    print(translate_prediction(result, labels, get_max=False))
    print(translate_prediction(result, labels, get_max=True))

# Predict more images
if (args['dataset']):
    print_info('Dataset prediction...')
    image_data, image_labels, labels_dict, path_list = DataLoader.load_scaled_data_with_labels(
        args['dataset'],
        image_width,
        image_height,
        correct_dataset_size=False
    )
    image_data = preproc.apply(image_data)

    test_training(image_data, image_labels, path_list, model, labels)
