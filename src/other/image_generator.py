from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import argparse
import cv2
import os



IMAGE_WIDTH = 50
IMAGE_HEIGHT = 50

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())


def image_variation(image_path):
    # Data generator
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    img = load_img(image_path)
    # Numpy array with shape (3, 150, 150)
    img_vector = img_to_array(img)
    # Numpy array with shape (1, 3, 150, 150)
    img_vector = img_vector.reshape((1, ) + img_vector.shape)

    num_of_images = 0
    for batch in datagen.flow(img_vector, batch_size=1, 
                            save_to_dir='preview', save_prefix='weapon',
                            save_format='jpeg'):
        num_of_images += 1
        if (num_of_images >= 20):
            break;

def image_resize(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    output_path = '../dataset/output_images/'
    image_name = 'zbran_'
    file_extention = '.png'
    image_counter = 0
    complete_path = ''
    while (True):
        complete_path = output_path + image_name + str(image_counter) + file_extention
        if (not os.path.exists(complete_path)):
            break;
        image_counter += 1

    cv2.imwrite(complete_path, image)

image_resize(args['image'])
