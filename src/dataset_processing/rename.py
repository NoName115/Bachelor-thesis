import re
import sys
import glob
import os

#EXTENSTION = '.png'
EXTENSTION = '.jpg\n'

#p0.0_y47.0_r0.0.jpg
if (len(sys.argv) != 2):
    print("Invalid input arguments")
    exit(-1)

images = []
paths = glob.glob(sys.argv[1] + "*" + EXTENSTION)

print(paths)

#r\d+\.\d+
for image in paths:
    indx = image.rfind('/') + 1
    image_path = image[indx:]

    # Pitch axis
    axis_p = re.search('p(-?\d+\.\d+)', image_path)
    axis_p = float(axis_p.groups()[0])
    
    if (axis_p < 0):
        axis_p += 360

    # Remove file
    if (axis_p > 90 and axis_p <= 270):
        print("removing " + image)
        os.remove(image)
        continue

    # Yaw axis
    axis_y = re.search('y(-?\d+\.\d+)', image_path)
    axis_y = float(axis_y.groups()[0])
    
    if (axis_y < 0):
        axis_y += 360

    # Remove file
    '''
    if (axis_y > 90 and axis_y <= 270):
        print("removing " + image)
        os.remove(image)
        continue
    '''

    # Roll axis
    axis_r = re.search('r(-?\d+\.\d+)', image_path)
    axis_r = float(axis_r.groups()[0])
    
    if (axis_r < 0):
        axis_r += 360

    '''
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

    new_path = "p" + str(axis_p) + '_y' + str(axis_y) + '_r' + str(axis_r) + '.jpg'#EXTENSTION

    if (image_path != new_path):
        print(image_path + ' --> ' + new_path)
        os.rename(image, image[:indx] + new_path)
