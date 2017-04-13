import glob
import os
import cPickle as pickle

BASE_DIR = "/home/rolan/CS-295/PA1"
IMAGE_DIR = BASE_DIR + "/cohn-kanade-images"

def count_subjects():
    subpath = os.path.join(IMAGE_DIR, "*")
    subjects = glob.glob(subpath)
    print "there are {} subjects".format(len(subjects))

def count_sequences():
    seqpath = os.path.join(IMAGE_DIR, "*", '*')
    sequences = glob.glob(seqpath)
    print "there are {} sequences".format(len(sequences))

def count_images():
    imgpath = os.path.join(IMAGE_DIR, "*", "*", "*")
    images = glob.glob(imgpath)
    print "there are {} images".format(len(images))

count_images()

EMO_DIR = BASE_DIR + "/ck+/Emotion"
def count_emofiles():
    emopath = os.path.join(EMO_DIR, "*", "*","*")
    emofiles = glob.glob(emopath)
    for f in emofiles:
        print f

    print "there are {} emofiles".format(len(emofiles))

def getemofromfile(f):
    fh = open(f)
    for l in fh.readlines():
        pass
    return int((float(l.strip())))

def emotags_to_dict():
    emopath = os.path.join(EMO_DIR, "*","*", "*")
    emofiles = glob.glob(emopath)
    emodict = {}
    for f in emofiles:
        todrop = len(EMO_DIR) +1
        newf = f[todrop:]
        pathnm= os.path.dirname(newf)
        fname =  os.path.basename(newf)
        emo = getemofromfile(f)
        emodict[pathnm] = emo
    print "There were {} emofiles".format(len(emodict.keys()))

    with open("emodict.pickle", 'wb') as fp:
        pickle.dump(emodict, fp)

emotags_to_dict()
