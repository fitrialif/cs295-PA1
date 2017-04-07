import glob
import os

BASE_DIR = "/home/rolan/CS-295/PA1"
IMAGE_DIR = BASE_DIR + "/cohn-kanade-images"


subpath = os.path.join(IMAGE_DIR, "*")
subjects = glob.glob(subpath)
print "there are {} subjects".format(len(subjects))

seqpath = os.path.join(IMAGE_DIR, "*", '*')
sequences = glob.glob(seqpath)
print "there are {} sequences".format(len(sequences))
