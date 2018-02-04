from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import argparse
import cv2
import os



IMAGE_WIDTH = 50
IMAGE_HEIGHT = 50

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    "--image",
    required=True,
	help="path to input image"
)
ap.add_argument(
    "--output",
    required=True,
	help="path to output directory"
)
args = vars(ap.parse_args())


def image_variation(image_path, output_folder):
    generate_images = 50

    # Data generator - 50 images (44)
    datagen = ImageDataGenerator(
        #featurewise_center=False,
        #samplewise_center=True,
        #featurewise_std_normalization=False,
        #samplewise_std_normalization=False,
        #zca_whitening=True,
        #zca_epsilon=1e-6,

        #rotation_range=180,
        #width_shift_range=0.2,  # 6 images - 0.2
        #height_shift_range=0.3,  # 8 images - 0.3
        #zoom_range=[0.7, 1.3],   # 10 images - [0.6, 1.4]
        #channel_shift_range=10,  # 10 images - 10
        fill_mode='nearest',
        #horizontal_flip=True,      # 5 images - True
        #vertical_flip=True,       # 5 images - True

        #shear_range=0.2,
    )

    datagen = ImageDataGenerator(
        rotation_range=180,
        width_shift_range=0.2,   # 6 images - 0.2
        height_shift_range=0.3,  # 8 images - 0.3
        zoom_range=[0.6, 1.4],   # 10 images - [0.6, 1.4]
        channel_shift_range=10,  # 10 images - 10
        fill_mode='nearest',
        horizontal_flip=True,    # 5 images - True
        vertical_flip=True,      # 5 images - True
    )

    # Check folder
    if (not os.path.exists(output_folder)):
        os.mkdir(output_folder)

    img = load_img(image_path)
    # Numpy array with shape (3, 150, 150)
    img_vector = img_to_array(img)
    # Numpy array with shape (1, 3, 150, 150)
    img_vector = img_vector.reshape((1, ) + img_vector.shape)

    num_of_images = 0
    for batch in datagen.flow(img_vector, batch_size=1,
                            save_to_dir=output_folder,
                            save_prefix='weapon',
                            save_format='jpeg'):
        num_of_images += 1
        if (num_of_images >= generate_images):
            break;

def image_resize(image_path, output_folder):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

    image_name = 'zbran_'
    file_extention = '.png'
    image_counter = 0
    complete_path = ''
    while (True):
        complete_path = output_folder + image_name + str(image_counter) + file_extention
        if (not os.path.exists(complete_path)):
            break;
        image_counter += 1

    cv2.imwrite(complete_path, image)

#image_resize(args['image'], args['output'])
image_variation(args['image'], args['output'])
