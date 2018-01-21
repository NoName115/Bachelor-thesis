from keras.models import load_model
from base import parse_arguments_training
from loader import DataLoader


IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

args = parse_arguments_training()

image = DataLoader.load_and_preprocess_image(args['image'])

model = load_model(args['model'])
