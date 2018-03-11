from keras.optimizers import Adam
from .preprocessing import Preprocessor, Preprocessing
from .base import parse_arguments_training, test_training
from .loader import DataLoader, DataSaver
from .models import LeNet, KerasBlog


EPOCHS = 1
BS = 16
IMAGE_WIDTH = 60
IMAGE_HEIGHT = 60

# --model, --dataset
args = parse_arguments_training()

# Load input images & split it
images, labels, labels_dict, path_list = DataLoader.load_scaled_data_with_labels(
     'dataset/training_data/',
     IMAGE_WIDTH,
     IMAGE_HEIGHT,
     correct_dataset_size=True
)

# Preprocessing
prepro = Preprocessor()
prepro.add_func(Preprocessing.normalize)
prepro.add_func(Preprocessing.grayscale)
prepro.add_func(Preprocessing.reshape)
images = prepro.apply(images)

# Datagen
prepro.set_datagen()

# Spliting data to training, validation & test
splited_data = DataLoader.split_data(images, labels, path_list)
train_x, train_y, train_p = splited_data[0:3]
val_x, val_y, val_p = splited_data[3:6]
test_x, test_y, test_p = splited_data[6:9]

# Building model
#model_class = LeNet(train_x.shape, labels_dict)
model_class = KerasBlog(train_x.shape, labels_dict)

'''
# Optimizer
INIT_LR = 1e-3
model_class.set_optimizer(
    Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
)
'''

# Training model
result = model_class.train(
    train_x, train_y,
    val_x, val_y,
    datagen=prepro.get_datagen(),
    epochs=EPOCHS,
    batch_size=BS,
)

DataSaver.save_model(args["model"], model_class, prepro, with_datetime=True)
test_training(test_x, test_y, test_p, model_class, labels_dict)
