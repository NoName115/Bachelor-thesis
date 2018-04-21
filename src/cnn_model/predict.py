from base import parse_arguments_prediction, Algorithm
from printer import print_info, print_error
from loader import DataLoader
from evaluation import evaluate_model


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
threshold = int(args['th']) if (args['th']) else 5

alg = model_class.algorithm
if (alg == Algorithm.CNN):
    alg += model_class.model_type

print_info("Model algorithm - " + alg)

# Predict one image
if (args['image']):
    print_info('Image prediction...')
    image = DataLoader.load_image(
        args['image'],
        image_width,
        image_height
    )

    image = image.reshape((1,) + image.shape)
    image = preproc.apply(image)

    evaluate_model(model_class, image, None, None, threshold)

# Predict more images
if (args['dataset']):
    print_info('Dataset prediction...')
    if (alg == Algorithm.CNN_A):
        image_data, image_labels, _, path_list = DataLoader.load_angle_images(
            args['dataset'],
            image_width,
            image_height,
            range(0, 360, 180)  # Not important
        )
    else:
        image_data, image_labels, _, path_list = DataLoader.load_images_from_folder(
            args['dataset'],
            image_width,
            image_height,
            labels_dict=model_class.labels_dict,
            correct_dataset_size=False
        )

    image_data = preproc.apply(image_data)

    evaluate_model(model_class, image_data, image_labels, path_list, threshold)

# Predict images from file
if (args['file']):
    print_info('File prediction...')
    if (alg == Algorithm.CNN_A):
        image_data, image_labels, path_list = DataLoader.load_images_from_file(
            args['file'],
            image_width,
            image_height,
            labels_dict=None
        )
    else:
        image_data, image_labels, _, path_list = DataLoader.load_images_from_file(
            args['file'],
            image_width,
            image_height,
            labels_dict=model_class.labels_dict
        )

    image_data = preproc.apply(image_data)

    evaluate_model(model_class, image_data, image_labels, path_list, threshold)
