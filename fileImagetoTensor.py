import tensorflow as tf

class CVDataToTensors():
    def getimages(imagelist):
        images = []
        for fname in imagelist:
            f = tf.read_file(fname)





if __name__ == '__main__':
