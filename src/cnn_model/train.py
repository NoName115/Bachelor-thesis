from keras.optimizers import Adam
from base import parse_arguments_training
from loader import DataLoader
from models import LeNet, KerasBlog


EPOCHS = 10
INIT_LR = 1e-3
BS = 16
IMAGE_WIDTH = 50
IMAGE_HEIGHT = 50

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
#model = LeNet(IMAGE_WIDTH, IMAGE_HEIGHT, labels_dict)
model = KerasBlog(IMAGE_WIDTH, IMAGE_HEIGHT, labels_dict)
'''
opt = Adam(
    lr=INIT_LR,
    decay=INIT_LR / EPOCHS
)
model.model.compile(
    loss='binary_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)
'''
model.model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

results = model.model.fit_generator(
    data_gen.flow(train_x, train_y, batch_size=BS),
    steps_per_epoch=2000 // BS, # steps_per_epoch=len(train_x) // BS,
    epochs=EPOCHS,
    validation_data=(test_x, test_y),
    validation_steps=800 // BS
)

model.save(args["model"])
