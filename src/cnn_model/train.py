from keras.optimizers import Adam, SGD
from preprocessing import Preprocessor, Preprocessing, AngleGenerator
from base import parse_arguments_training
from printer import print_error
from loader import DataLoader, DataSaver
from models import LeNet, KerasBlog, MyModel, VGG16


EPOCHS = 5#45
BS = 16
IMAGE_WIDTH = 16
IMAGE_HEIGHT = 16
MODEL_NAME = 'MyModel_Angle'

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
        range(0, 360),
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
model_class = MyModel(train_x.shape, labels_dict, model_name=MODEL_NAME)
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
        loss='binary_crossentropy',
        optimizer='adam',
        #optimizer='rmsprop',
        metrics=['accuracy']
    )
    model_class.model.fit_generator(
        AngleGenerator(
            train_x,
            labels_dict,
            BS
        ),
        epochs=EPOCHS,
        validation_data=AngleGenerator(
            val_x,
            labels_dict,
            BS
        )
    )

DataSaver.save_model(args["model"], model_class, prepro, with_datetime=True)
#model_class.evaluate(test_x, test_y, test_p)
