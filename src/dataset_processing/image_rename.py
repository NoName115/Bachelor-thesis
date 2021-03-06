# Script for renaming images
#
# Author: Robert Kolcun, FIT
# <xkolcu00@stud.fit.vutbr.cz>

import re
import sys
import glob
import os

#EXTENSTION = '.png'
EXTENSTION = '.jpg'

#p0.0_y47.0_r0.0.jpg
if (len(sys.argv) != 2):
    print("Invalid input arguments")
    exit(-1)

images = []
paths = glob.glob(sys.argv[1] + "*" + EXTENSTION)

print("Number of paths: " + str(len(paths)) + " in " + sys.argv[1])

#r\d+\.\d+
for image in paths:
    indx = image.rfind('/') + 1
    image_path = image[indx:]

    try:
        image_lastname = image_path[image_path.find('-'):]    # find lastname
        image_lastname = image_lastname[:-len(EXTENSTION)]      # remove extension
    except Exception:
        raise
        print("Error")
        continue

    print("LN: " + image_lastname)

    # Pitch axis
    axis_p = re.search('p(-?\d+\.\d+)', image_path)
    axis_p = float(axis_p.groups()[0])

    '''
    if (axis_p < 0):
        axis_p += 360

    # Remove file
    if (axis_p > 90 and axis_p <= 270):
        print("removing " + image)
        os.remove(image)
        continue
    '''

    # Yaw axis
    axis_y = re.search('y(-?\d+\.\d+)', image_path)
    axis_y = float(axis_y.groups()[0])
    
    '''
    if (axis_y < 0):
        axis_y += 360

    # Remove file
    if (axis_y > 90 and axis_y <= 270):
        print("removing " + image)
        os.remove(image)
        continue
    '''

    # Roll axis
    axis_r = re.search('r(-?\d+\.\d+)', image_path)
    axis_r = int(float(axis_r.groups()[0]))
    
    if (axis_r != 0):
        continue

    '''
    if (axis_r < 0):
        axis_r += 360

    # Remove file
    if (axis_r > 90 and axis_r <= 270):
        print("removing " + image)
        os.remove(image)
        continue
    '''

    '''
    if (int(axis_r) != 0):
        axis_p = axis_r
        axis_r = 0.0
    '''

    #new_path = "p" + str(axis_p) + '_y' + str(axis_y) + '_r' + str(axis_r) + EXTENSTION
    new_path = "p" + str(axis_r) + '_y' + str(axis_y) + '_r' + str(axis_p) + image_lastname + EXTENSTION

    if (image_path != new_path):
        print(image_path + ' --> ' + new_path)
        os.rename(image, image[:indx] + new_path)
