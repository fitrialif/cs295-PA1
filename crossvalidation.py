
import glob
import os

DATA_DIR = "/Volumes/Data/Dropbox/PhD-CS/CS 295 -Deep Learning/PA1/data/resized"
class NFoldCV(object):
    def __init__(self, folds):
        self.folds = 10
        file_path = os.path.join(DATA_DIR, "*")
        self.files = glob.glob(file_path)
        self.count = len(self.files)

    def getNextBatch(self):
        pass

    def getStats(self):
        print "File Count is {}. Some files are listed below:".format(self.count)
        for i,f in enumerate(self.files[1:10]):
            print "{}: {}".format(i,f)


if __name__ == '__main__':
    cx = NFoldCV(10)
    cx.getStats()