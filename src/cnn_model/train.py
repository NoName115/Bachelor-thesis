from keras.optimizers import Adam, SGD
from preprocessing import Preprocessor, Preprocessing, AngleGenerator, \
                          ClassificationGenerator
from base import parse_arguments_training, angle_error, Algorithm
from printer import print_error
from loader import DataLoader, DataSaver
from models import *
from evaluation import evaluate_model


#  --dataset --model --alg [--ep] [--bs] [--rt]
args = parse_arguments_training()

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
MODEL_NAME = args['alg']

ROTATE_ANGLE = 5
if (MODEL_NAME == Algorithm.CNN_A):
    ROTATION_TYPE = args['rt']  # yaw, pitch, roll

EPOCHS = 45 if (not args['ep']) else int(args['ep'])
BS = 16 if (not args['bs']) else int(args['bs'])

alg = Algorithm.translate(args['alg'])


# Load input data
if (alg == Algorithm.CNN_A):
    images, labels, labels_dict, path_list = DataLoader.load_angle_images(
        args['dataset'],
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        range(0, 360, ROTATE_ANGLE),
        angle_type=[ROTATION_TYPE]
    )
    MODEL_NAME += '_' + ROTATION_TYPE + "-" + str(ROTATE_ANGLE)
else:
    images, labels, labels_dict, path_list = DataLoader.load_images_from_folder(
        args['dataset'],
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        correct_dataset_size=True,
    )

# Using keras or scikit-learn
using_keras = False
if (alg == Algorithm.CNN_A or alg == Algorithm.CNN_C):
    using_keras = True

# Preprocessing
preproc = Preprocessor()
preproc.set_datagen()

if (using_keras):
    preproc.add_func(Preprocessing.normalize)
    images = preproc.apply(images)
else:
    preproc.add_func(Preprocessing.normalize)
    preproc.add_func(Preprocessing.flat)

# Split data only into training & test
splited_data = DataLoader.split_data(
    images, labels, path_list,
    num_classes=len(labels_dict),
    split_size=0.3,
    use_to_categorical=True if (using_keras) else False,
    get_test=True if (using_keras) else False,
)

train_x, train_y, train_p = splited_data[0:3]
if (using_keras):
    val_x, val_y, val_p = splited_data[3:6]
    test_x, test_y, test_p = splited_data[6:9]
else:
    test_x, test_y, test_p = splited_data[3:6]

# Data augmentation
if (not using_keras):
    train_x, train_y, train_p = ClassificationGenerator(
        train_x, train_y, train_p,
        1, preproc.get_datagen()
    ).flow()
    train_x = preproc.apply(train_x)

# Build & train model

'''
# Best options
    - no dropout
    - dropout with more epochs >40
    - rmsprop/adam optimizer
'''

history = None
if (alg == Algorithm.CNN_C):
    model_class = MyModel(
        train_x.shape, labels_dict, 'class',
        model_name=MODEL_NAME
    ).build()
    history = model_class.train(
        train_x, train_y,
        val_x, val_y,
        datagen=preproc.get_datagen(),
        epochs=EPOCHS,
        batch_size=BS,
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )
elif (alg == Algorithm.CNN_A):
    model_class = MyModel(
        train_x.shape, labels_dict, 'angle',
        model_name=MODEL_NAME, rotation_type=ROTATION_TYPE
    ).build()
    history = model_class.train(
        train_x, train_y,
        val_x, val_y,
        datagen=AngleGenerator(labels_dict, ROTATION_TYPE),
        epochs=EPOCHS,
        batch_size=BS,
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=[angle_error]
    )
elif (alg == Algorithm.SVM):
    model_class = SVM(
        [None, IMAGE_WIDTH, IMAGE_HEIGHT, 3], labels_dict, 'class',
        model_name=MODEL_NAME
    ).build()
    model_class.train(
        train_x, train_y
    )
elif (alg == Algorithm.KMEANS):
    model_class = KMeans(
        [None, IMAGE_WIDTH, IMAGE_HEIGHT, 3], labels_dict, 'class',
        model_name=MODEL_NAME
    ).build()
    model_class.train(
        train_x, train_y
    )
elif (alg == Algorithm.MLP):
    model_class = MLPerceptron(
        [None, IMAGE_WIDTH, IMAGE_HEIGHT, 3], labels_dict, 'class',
        model_name=MODEL_NAME
    ).build(
        solver='lbfgs', alpha=1e-5,
        hidden_layer_sizes=(50,), random_state=1
    )
    model_class.train(
        train_x, train_y
    )
else:
    print_error("Unknown algorithm")

# Save trained model
DataSaver.save_model(
    args['model'],
    model_class, preproc,
    training_history=history,
    with_datetime=True
)

# Evaluate trained model
if (not using_keras):
    test_x, test_y, test_p = ClassificationGenerator(
        test_x, test_y, test_p,
        1, preproc.get_datagen()
    ).flow()
    test_x = preproc.apply(test_x)

evaluate_model(model_class, test_x, test_y, test_p)
