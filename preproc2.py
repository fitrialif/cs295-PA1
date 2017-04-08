import glob
import os
import cPickle as pickle
import shutil

BASE_DIR = "/home/rolan/CS-295/PA1"
IMAGE_DIR = BASE_DIR + "/cohn-kanade-images"
DATA_DIR = BASE_DIR + "/data"

def cpandrenameimages():
    with open("emodict.pickle", 'rb') as fp:
        emodict = pickle.load(fp)

    for key in emodict.keys():
        subpath = IMAGE_DIR + "/" + key
        files = glob.glob(os.path.join(subpath, "*"))
        for f in files:
            fname = os.path.basename(f)
            newfile = "E" + str(emodict[key]) + "--" + fname.split('.')[0] + "." + (fname.split('.')[1]).strip()
            shutil.copyfile(f, os.path.join(DATA_DIR, newfile))




cpandrenameimages()



