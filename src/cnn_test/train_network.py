# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
 
# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from cnn_test import LeNet
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os


EPOCHS = 35
INIT_LR = 1e-3
BS = 32
IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28

# data labels
print("[INFO] Loading images...")
data = []
labels = []
args = {}

args['dataset'] = "../../dataset/training_data/"
args['model'] = "dlhe_kratke_zbrane.model"
args['plot'] = "plot.png"

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
	# load the image, pre-process it, and store it in the data list
	image = cv2.imread(imagePath)
	image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
	image = img_to_array(image)
	data.append(image)
 
	# extract the class label from the image path and update the
	# labels list
	label = imagePath.split(os.path.sep)[-2]
	label = 1 if label == "long_weapon" else 0
	labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)
 
# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

print(trainY)
print(testY)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=2)
testY = to_categorical(testY, num_classes=2)

print(trainY)
print(testY)
exit()

# construct the image generator for data augmentation
aug = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.1,
	height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
	horizontal_flip=True,
    fill_mode="nearest"
)

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(
	width=IMAGE_WIDTH,
	height=IMAGE_HEIGHT,
	depth=3,
	classes=2
)
opt = Adam(
	lr=INIT_LR,
	decay=INIT_LR / EPOCHS
)
model.compile(
	loss="binary_crossentropy",
	optimizer=opt,
	metrics=["accuracy"]
)
 
# train the network
print("[INFO] training network...")
H = model.fit_generator(
	aug.flow(trainX, trainY, batch_size=BS),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // BS,
	epochs=EPOCHS,
	verbose=1
)

# save the model to disk
print("[INFO] serializing network...")
model.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on Santa/Not Santa")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
