from keras.models import load_model
from base import parse_arguments_prediction, translate_prediction
from loader import DataLoader


args = parse_arguments_prediction()

model_data = DataLoader.load_model_data(
    args['model']
)

model = model_data[0]
image_width = model_data[1]
image_height = model_data[2]
labels = model_data[3]
preproc = model_data[4]
datagen = model_data[5]

image = DataLoader.load_and_preprocess_image(
    args['image'],
    image_width,
    image_height,
    preproc
)

result = model.predict(image)[0]
print(translate_prediction(result, labels, get_max=True))
