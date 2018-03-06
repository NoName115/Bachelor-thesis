from keras.optimizers import Adam
from base import parse_arguments_training, test_training
from loader import DataLoader, DataSaver
from models import LeNet, KerasBlog
from preprocessing import Preprocessor, Preprocessing


EPOCHS = 1
INIT_LR = 1e-3
BS = 16
IMAGE_WIDTH = 50
IMAGE_HEIGHT = 50

# --model, --image
args = parse_arguments_training()

# Load input images & split it
images, labels, labels_dict = DataLoader.load_scaled_data_with_labels(
     'dataset/training_data/',
     IMAGE_WIDTH,
     IMAGE_HEIGHT
)

splited_data = DataLoader.split_data(images, labels)
train_x, train_y = splited_data[0:2]
val_x, val_y = splited_data[2:4]
test_x, test_y = splited_data[4:6]

# Preprocessing
prepro = Preprocessor()
prepro.add_func(Preprocessing.normalize)
prepro.add_func(Preprocessing.grayscale)
prepro.add_func(Preprocessing.reshape)
train_x = prepro.apply(train_x)
val_x = prepro.apply(val_x)

prepro.set_datagen()


# Building model
#model = LeNet(IMAGE_WIDTH, IMAGE_HEIGHT, labels_dict)
model = KerasBlog(IMAGE_WIDTH, IMAGE_HEIGHT, labels_dict, depth=1)
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
model.model.compile(
    loss='binary_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

results = model.model.fit_generator(
    prepro.get_datagen().flow(train_x, train_y, batch_size=BS),
    steps_per_epoch=len(train_x) // BS,
    epochs=EPOCHS,
    validation_data=(val_x, val_y),
    validation_steps=len(val_x) // BS
)

#DataSaver.save_model(args["model"], model, prepro)

test_training(test_x, test_y, model.model, labels_dict)
