import tensorflow as tf
import os
import numpy as np
import cv2
def filestoTFdata(files):
    '''
    
    :param files:is a list of filenames of format Ex-xxxxxxx.png representing images of emotions 
    :return: list of tensors representing the images in the files 
    '''
    images= []
    labels = []
    for f in files:
        fn = os.path.basename(f)
        emo = int(fn.split('_')[0][1]) - 1
        onehot = np.array([int(i==emo) for i in range(7)], dtype=np.uint8)
        emo_label = np.reshape(onehot, (1, 7))
        labels.append(emo_label)
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        expanded_img = np.expand_dims(np.expand_dims(img,0),3)
        image = np.array(expanded_img)
        images.append(image)

    return images, labels


if __name__== "__main__":
    filelist = ['../data/resized/E3--S060_005_00000011.png']
    images, labels = filestoTFdata(filelist)