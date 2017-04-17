import tensorflow as tf
import os
import numpy as np

def filestoTensors(files):
    '''
    
    :param files:is a list of filenames of format Ex-xxxxxxx.png representing images of emotions 
    :return: list of tensors representing the images in the files 
    '''
    imagetensors = []
    imagelabels = []
    for f in files:
        fn = os.path.basename(f)
        emo = int(fn.split('_')[0][1]) - 1
        onehot = np.array([int(i==emo) for i in range(7)])
        emo_label = tf.expand_dims(tf.expand_dims(tf.convert_to_tensor(onehot, dtype=tf.float32), 0), 0)
        imagelabels.append(emo_label)
        imagefile = tf.read_file(f)
        tensor = tf.image.decode_png(imagefile, channels=1)
        resized_tensor = tf.expand_dims(tf.image.resize_images(tensor, (224,224)), 0)
        imagetensors.append(resized_tensor)
    return imagetensors, imagelabels


if __name__== "__main__":
    filelist = ['../data/resized/E3-S060_005_000000011.png']
    imagetensors, imagelabels = filestoTensorss(filelist)