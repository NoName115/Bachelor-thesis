from models import LeNet
from loader import DataLoader
from base import parse_arguments_training, parse_arguments_prediction


EPOCHS = 35
INIT_LR = 1e-3
BS = 32
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

parse_arguments_training()

# Load input images
images, labels, labels_leg, num_of_lab = DataLoader.load_scaled_data_with_labels(
     'dataset/training_data/',
     IMAGE_WIDTH,
     IMAGE_HEIGHT
)
train_x, train_y, test_x, test_y = split_data(images, labels)
data_gen = DataLoader.get_default_datagen()

# Building model
model = LeNet.build(IMAGE_WIDTH, IMAGE_HEIGHT, 2)
opt = Adam(
    lr=INIT_LR,
    decay=INIT_LR / EPOCHS
)
model.compile(
    loss='binary_crossentropy',
    optimazer=opt,
    metrics=['accuracy']
)

results = model.fit_generator(
    aug.flow(train_x, train_y, batch_size=BS),
    validation_data=(test_x, test_y),
    steps_per_epoch=len(train_x) // BS,
    epochs=EPOCHS
)

model.save(args["model"])
