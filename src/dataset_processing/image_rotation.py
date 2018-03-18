import argparse
import cv2
import glob


# Input dataset
ap = argparse.ArgumentParser()
ap.add_argument(
    "--dataset",
    required=True,
    help="path to input data for training",
    )
args = vars(ap.parse_args())

images_to_rotate = glob.glob(args['dataset'] + '*')

for image_path in images_to_rotate:
    image_name = image_path[image_path.rfind('/') + 1: ]

    # show the output image
    img = cv2.imread(image_path)
    cv2.imshow("Image", img)
    key = cv2.waitKey(0)

    if (key == ord('r')):
        cv2.imwrite(args['dataset'] + image_name, cv2.flip( img, 1 ))
        print("Rotated, saved - " + image_name)
    else:
        print("OK")

    cv2.destroyAllWindows()
