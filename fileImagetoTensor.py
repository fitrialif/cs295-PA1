from crossvalidation import NFoldCV
import tensorflow as tf

class CVDataToTensors():
    def getimages(self, imagelist):
        images = []
        for fname in imagelist:
            imagefile = tf.read_file(fname)
            tensor = tf.image.decode_png(imagefile, channels=1)
            resized_tensor = tf.expand_dims(tf.image.resize_images(tensor, (224, 224)), 0, name="input")
            print "{}".format(resized_tensor)
            images.append(resized_tensor)
        return images


if __name__ == '__main__':
    u = CVDataToTensors()
    cx = NFoldCV(10)
    cx.getStats()
    cx.showfolds()
    _, tset = cx.getBatch(1)
    images = u.getimages(tset)
