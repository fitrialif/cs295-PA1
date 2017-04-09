import cv2
import glob
import os


BASE_DIR = "/home/rolan/CS-295/PA1"
IMAGE_DIR = BASE_DIR + "/cohn-kanade-images"
DATA_DIR = BASE_DIR + "/data"
RESIZED_DIR = DATA_DIR + "/resized"

files_pattern = "*.png"
file_list = glob.glob(os.path.join(DATA_DIR, files_pattern))

for idx,f in enumerate(file_list):
    todrop = len(DATA_DIR)+1
    filename = f[todrop:]
    image=cv2.imread(f)
    resized_img = cv2.resize(image,(224,224), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(RESIZED_DIR, filename), resized_img)
    print "{} finished".format(idx)