from base import parse_arguments_training
from loader import DataLoader, DataSaver
from preprocessing import Preprocessor, Preprocessing, ClassificationGenerator
from printer import print_info
from models import *
from evaluation import evaluate_model


args = parse_arguments_training()

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128


from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics, cluster, datasets
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

images, labels, labels_dict, path_list = DataLoader.load_images_from_folder(
    args['dataset'],
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    correct_dataset_size=True,
)

# Split data only into training & test
splited_data = DataLoader.split_data(
    images, labels, path_list,  # Fix path_list size due to ClassGenerator
    num_classes=len(labels_dict),
    split_size=0.3,
    use_to_categorical=False,
    get_test=False,
)
train_x, train_y, train_p = splited_data[0:3]
val_x, val_y, val_p = splited_data[3:6]

preproc = Preprocessor()
preproc.add_func(Preprocessing.normalize)
preproc.add_func(Preprocessing.flat)
preproc.set_datagen()

# Data augmentation
train_x, train_y, train_p = ClassificationGenerator(
    train_x, train_y, train_p,
    1, preproc.get_datagen()
).flow()

train_x = preproc.apply(train_x)

k_means = cluster.KMeans(n_clusters=len(labels_dict))
k_means.fit(train_x)

correct = 0
incorrect = 0

for predict, expect in zip(k_means.labels_, train_y):
    if (predict == expect):
        print("Correct")
        correct += 1
    else:
        print("Incorrect")
        incorrect += 1

print("C: " + str(correct))
print("I: " + str(incorrect))

#print(k_means.labels_.shape)

#print(k_means.labels_[::10])
#print(labels[::10])

exit()











if (args['type'] == 'train'):
    images, labels, labels_dict, path_list = DataLoader.load_images_from_folder(
        args['dataset'],
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        correct_dataset_size=True,
    )

    # Split data only into training & test
    splited_data = DataLoader.split_data(
        images, labels, path_list,  # Fix path_list size due to ClassGenerator
        num_classes=len(labels_dict),
        split_size=0.3,
        use_to_categorical=False,
        get_test=False,
    )
    train_x, train_y, train_p = splited_data[0:3]
    val_x, val_y, val_p = splited_data[3:6]

    preproc = Preprocessor()
    preproc.add_func(Preprocessing.normalize)
    preproc.add_func(Preprocessing.flat)
    preproc.set_datagen()

    # Data augmentation
    train_x, train_y, train_p = ClassificationGenerator(
        train_x, train_y, train_p,
        1, preproc.get_datagen()
    ).flow()

    train_x = preproc.apply(train_x)

    model_class = SVM(
        [None, IMAGE_WIDTH, IMAGE_HEIGHT, 3], labels_dict, args['type'],
        model_name="SVM"
    )
    model_class.train(
        train_x, train_y
    )

    DataSaver.save_model(
        args['model'],
        model_class, preproc,
        training_history=None,
        with_datetime=True
    )

    '''
    # Evaluation process
    val_x, val_y, val_p = ClassificationGenerator(
        val_x, val_y, val_p,
        1, preproc.get_datagen()
    ).flow()
    # Normlize & flat images
    val_x = preproc.apply(val_x)
    # Evaluate model
    evaluate_model(model_class, val_x, val_y, val_p)
    '''
else:
    print_info("Testing model, 2.option")

    '''
    model_class, preproc = DataLoader.load_model_data(args['model'])
    images, labels, _, path_list = DataLoader.load_images_from_folder(
        args['dataset'],
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        correct_dataset_size=True,
    )
    # Data augmentation
    train_x, train_y, train_p = ClassificationGenerator(
        images, labels, path_list,
        2, preproc.get_datagen()
    ).flow()

    train_x = preproc.apply(train_x)

    evaluate_model(model_class, train_x, train_y, train_p)
    '''
