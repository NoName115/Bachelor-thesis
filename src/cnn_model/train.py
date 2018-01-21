from keras.optimizers import Adam
from base import parse_arguments_training
from loader import DataLoader
from models import LeNet


EPOCHS = 5 #25
INIT_LR = 1e-3
BS = 32
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

# --model, --image
args = parse_arguments_training()

# Load input images
images, labels, labels_dict = DataLoader.load_scaled_data_with_labels(
     'dataset/training_data/',
     IMAGE_WIDTH,
     IMAGE_HEIGHT
)
train_x, train_y, test_x, test_y = DataLoader.split_data(images, labels)
data_gen = DataLoader.get_default_datagen()

# Building model
model = LeNet(IMAGE_WIDTH, IMAGE_HEIGHT, labels_dict)
opt = Adam(
    lr=INIT_LR,
    decay=INIT_LR / EPOCHS
)
model.model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

results = model.model.fit_generator(
    data_gen.flow(train_x, train_y, batch_size=BS),
    validation_data=(test_x, test_y),
    steps_per_epoch=len(train_x) // BS,
    epochs=EPOCHS
)

model.save(args["model"])
