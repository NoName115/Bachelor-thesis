from base import parse_arguments_training
from loader import DataLoader
from preprocessing import Preprocessor, Preprocessing

#'''
args = parse_arguments_training()

EPOCHS = 45
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128

images, labels, labesl_dict, path_list = DataLoader.load_images_from_folder(
    args['dataset'],
    IMAGE_WIDTH,
    IMAGE_HEIGHT,
    correct_dataset_size=True,
)

prepro = Preprocessor()
prepro.add_func(Preprocessing.normalize)
images = prepro.apply(images)

splited_data = DataLoader.split_data(
    images, labels, path_list,
    num_classes=len(labesl_dict),
    split_size=0.3,
    use_to_categorical=False,
)
train_x, train_y, train_p = splited_data[0:3]
val_x, val_y, val_p = splited_data[3:6]
test_x, test_y, test_p = splited_data[6:9]

# Flat image data to features
train_x = train_x.reshape((len(train_x), -1))
val_x = val_x.reshape((len(val_x), -1))

# Learning process
from sklearn import datasets, svm, metrics

classifier = svm.SVC(gamma=0.001)
classifier.fit(train_x, train_y)

# Evaluation process
expected = val_y
predicted = classifier.predict(val_x)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))

#'''

'''
from sklearn import datasets, svm, metrics

import matplotlib.pyplot as plt

# The digits dataset
digits = datasets.load_digits()

#print(digits.shape)

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

#print(digits.images.shape)
#print(data.shape)

#print("--------------")
#print(digits.target.shape)
#print(digits.target)

#exit()

# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples // 2], digits.target[:n_samples // 2])

# Now predict the value of the digit on the second half:
expected = digits.target[n_samples // 2:]
predicted = classifier.predict(data[n_samples // 2:])

print(expected.shape)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()
'''
