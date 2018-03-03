# kill $(ps -A | grep python | awk '{print $1}')

import argparse
import cv2
import glob
import shutil


ap = argparse.ArgumentParser()
ap.add_argument(
    "--images",
    required=True,
	help="path to unsorted images"
)
args = vars(ap.parse_args())

'''
folders_to_load = glob.glob(
    args['images'] + '*'
)
'''
folder_long_scene = 'filtered_dataset/long_scene/'
folder_long_weapon = 'filtered_dataset/long_weapon/'
folder_short_scene = 'filtered_dataset/short_scene/'
folder_short_weapon = 'filtered_dataset/short_weapon/'
folder_other = 'filtered_dataset/other/'

#print("Folder to load: " + '\n'.join(folders_to_load))

#for folder in folders_to_load:
#    images_to_load = glob.glob(folder + '/*.jpg')
#    print('Processing: ' + str(folder))

images_to_load = glob.glob(args['images'] + '*.jpg')
print(images_to_load)

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
    else:  # other
        cv2.imwrite(folder_other + image_name, img)
        print("Other")

    cv2.destroyAllWindows()

    '''
    # Remove folder
    print("Removing: " + folder)
    shutil.rmtree(folder, ignore_errors=True)
    '''
