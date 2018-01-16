import argparse
import cv2
import glob
import shutil


print(glob.glob("/home/adam/*.txt"))

'''
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
'''

folders_to_load = glob.glob(
    '../dataset/move_images_2/*'
)
folder_long_scene = '../dataset/long_scene/'
folder_long_weapon = '../dataset/long_weapon/'
folder_short_scene = '../dataset/short_scene/'
folder_short_weapon = '../dataset/short_weapon/'
folder_other = '../dataset/other/'

for folder in folders_to_load:
    images_to_load = glob.glob(folder + '/*.jpg')
    print(folder)

    for image_path in images_to_load:
        image_name = image_path[image_path.rfind('/') + 1: ]

        # show the output image
        img = cv2.imread(image_path)
        cv2.imshow("Image", img)
        key = cv2.waitKey(0)

        if (key == ord('q')):   # long_scene
            cv2.imwrite(folder_long_scene + image_name, img)
            print("Q")
        elif (key == ord('w')): # long_weapon
            cv2.imwrite(folder_long_weapon + image_name, img)
            print("W")
        elif (key == ord('e')): # short_scene
            cv2.imwrite(folder_short_scene + image_name, img)
            print("E")
        elif (key == ord('r')): # short_weapon
            cv2.imwrite(folder_short_weapon + image_name, img)
            print("R")
        elif (key == ord('o')):  # other
            cv2.imwrite(folder_other + image_name, img)
            print("O")
        else:
            print("BLABAL")

        cv2.destroyAllWindows()

    # Remove folder
    print("Rmoving: " + folder)
    shutil.rmtree(folder, ignore_errors=True)
