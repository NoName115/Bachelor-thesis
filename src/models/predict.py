# Script for prediction of image type and orientation
#
# Author: Róbert Kolcún, FIT
# <xkolcu00@stud.fit.vutbr.cz>

from base import parse_arguments_prediction
from loader import DataLoader
from evaluation import get_prediction


def get_max(results):
    return max(results, key=results.get)

# --class, --anglep, --angler, --angley, --image
args = parse_arguments_prediction()

# Load all models
model_class, prepro_class = DataLoader.load_model_data(args['class'])
model_angle_x, prepro_x = DataLoader.load_model_data(args['anglep'])
model_angle_y, prepro_y = DataLoader.load_model_data(args['angler'])
model_angle_z, prepro_z = DataLoader.load_model_data(args['angley'])

# Load & Reshape image
image = DataLoader.load_image(
    args['image'],
    model_class.width, model_class.height
)
image = image.reshape((1,) + image.shape)

# Predict weapon type and angle
weapon_type = get_prediction(model_class, prepro_class.apply(image))
angle_x = get_prediction(model_angle_x, prepro_x.apply(image))
angle_y = get_prediction(model_angle_y, prepro_y.apply(image))
angle_z = get_prediction(model_angle_z, prepro_z.apply(image))

print("\nType: {0}\tOrientation: {1}-{2}-{3} (pitch, roll, yaw)\n".format(
    get_max(weapon_type),
    get_max(angle_x), get_max(angle_y), get_max(angle_z)
))
