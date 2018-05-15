# Script for downloading images
#
# Author: Robert Kolcun, FIT
# <xkolcu00@stud.fit.vutbr.cz>

from glob import glob
import urllib.request


def download_images():
    counter = 0
    file_list = glob("dataset/*.txt")
    for file in file_list:
        with open(file, 'r') as open_file:
            for line in open_file:
                try:
                    print(line)
                    urllib.request.urlretrieve(
                        line,
                        'images/weapon_' + str(counter) + ".jpg"
                    )
                    counter += 1
                except KeyboardInterrupt:
                    continue

                except Exception as err:
                    print("ERR: " + str(err))

download_images()
