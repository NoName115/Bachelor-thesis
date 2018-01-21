from keras.models import load_model
from base import parse_arguments_prediction, translate_prediction
from loader import DataLoader


args = parse_arguments_prediction()

model, image_width, image_height, labels = DataLoader.load_model_data(
    args['model']
)
image = DataLoader.load_and_preprocess_image(
    args['image'],
    image_width,
    image_height
)

result = model.predict(image)[0]
print(translate_prediction(result, labels, get_max=True))
