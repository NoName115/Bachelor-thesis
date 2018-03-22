from keras.optimizers import Adam, SGD
from preprocessing import Preprocessor, Preprocessing, AngleGenerator
from base import parse_arguments_training, evaluate_model
from printer import print_error
from loader import DataLoader, DataSaver
from models import LeNet, KerasBlog, MyModel, VGG16


EPOCHS = 35 #45
BS = 32
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
MODEL_NAME = 'MyModel_Angle'
ROTATE_ANGLE = 18

# --model, --dataset
args = parse_arguments_training()

# Load input images & split it
if (args['type'] == "class"):
    images, labels, labels_dict, path_list = DataLoader.load_images_from_folder(
        args['dataset'],
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        correct_dataset_size=True,
    )
elif (args['type'] == "angle"):
    images, labels, labels_dict, path_list = DataLoader.load_angle_images(
        args['dataset'],
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        range(0, 360, ROTATE_ANGLE),
    )
else:
    print_error("Invalid input argument - 'type'")

# Preprocessing
prepro = Preprocessor()
prepro.add_func(Preprocessing.normalize)
#prepro.add_func(Preprocessing.grayscale)
#prepro.add_func(Preprocessing.reshape)
images = prepro.apply(images)

# Datagen
prepro.set_datagen()

# Spliting data to training, validation & test
splited_data = DataLoader.split_data(
    images, labels, path_list,
    num_classes=len(labels_dict), split_size=0.3
)
train_x, train_y, train_p = splited_data[0:3]
val_x, val_y, val_p = splited_data[3:6]
test_x, test_y, test_p = splited_data[6:9]

# Building model
#model_class = KerasBlog(train_x.shape, labels_dict)
#model_class = LeNet(train_x.shape, labels_dict)
model_class = MyModel(
    train_x.shape, labels_dict, args['type'],
    model_name=MODEL_NAME
)
#model_class = VGG16(train_x.shape, labels_dict)

'''
# Best options
    - no dropout
    - dropout with more epochs >40
    - rmsprop/adam optimizer
'''

#model_class.set_optimizer('sgd')

# Training model
if (args['type'] == "class"):
    model_class.train(
        train_x, train_y,
        val_x, val_y,
        datagen=prepro.get_datagen(),
        epochs=EPOCHS,
        batch_size=BS,
    )
else:
    model_class.model.compile(
        loss='categorical_crossentropy', #'binary_crossentropy',
        optimizer='adam',
        #optimizer='rmsprop',
        #optimizer=sgd,
        metrics=['categorical_accuracy']
    )
    model_class.model.fit_generator(
        AngleGenerator(labels_dict).flow(train_x, BS),
        steps_per_epoch=len(train_x) // BS,
        epochs=EPOCHS,
        validation_data=AngleGenerator(labels_dict).flow(val_x, BS),
        validation_steps=len(val_x) // BS
    )

DataSaver.save_model(
    args["model"],
    model_class, prepro,
    with_datetime=True
)
#model_class.evaluate(test_x, test_y, test_p)

evaluate_model(model_class, test_x, test_y, test_p)
