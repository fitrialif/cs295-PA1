import utils
import glob
import os
import random
import time

DATA_DIR = "../data/resized"


class NFoldCV(object):
    def __init__(self, folds):
        self.numfolds = 10
        file_path = os.path.join(DATA_DIR, "*")
        self.files = glob.glob(file_path)
        self.count = len(self.files)

        random.shuffle(self.files)
        skip = self.numfolds
        self.folds = [self.files[i::skip] for i in xrange(self.numfolds)]
        self.trainsets = [[file for file in self.files if file not in self.folds[j]] for j in range(len(self.folds))]


    def showfolds(self):
        for i in xrange(self.numfolds):
            fold = self.folds[i]
            trainset= self.trainsets[i]
            print "Fold {}: length is:{}".format(i, len(fold))
            print "Trainset {}: length is:{}".format(i, len(trainset))
            assert(len(fold)+len(trainset) == self.count)


    def getBatch(self,index):
        # self.folds[index] contains the filenames for the validation set
        # self.trainsets[index] contains the rest of the dataset filenames(less the validation set)
        vdata, vlabels = utils.filestoTensors(self.folds[index])
        tdata, tlabels = utils.filestoTensors(self.trainsets[index])
        return vdata, vlabels, tdata, tlabels

    def getStats(self):
        print "File Count is {}. Some files are listed below:".format(self.count)
        for i,f in enumerate(self.files[1:10]):
            print "{}: {}".format(i,f)


if __name__ == '__main__':
    cx = NFoldCV(10)
    cx.getStats()
    cx.showfolds()
    print "Start:{}".format(time.strftime("%H:%M:%S"))
    vdata, vlabels, tdata, tlabels = cx.getBatch(1)
    print "vdata_len {} vlabels_len {} tdata_len {} tlabels_len {}".format(len(vdata), len(vlabels), len(tdata), len(tlabels))
    print "End:{}".format(time.strftime("%H:%M:%S"))
    print "done"
