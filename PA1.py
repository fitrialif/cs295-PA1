import deXpression as dX
import crossvalidation as CV

import tensorflow as tf

x = tf.placeholder(tf.int8, shape=[])
y_ = tf.placeholder(tf.int8, shape=[])
conv1 = dX.conv1(x)
pool1 = dX.pool1(conv1)
lrn1 = dX.lrn1(pool1)
conv2a = dX.conv2a(lrn1)
pool2a = dX.pool2a(lrn1)
conv2b = dX.conv2b(pool2a)
conv2c = dX.conv2c(pool2a)
concat2 = dX.concat2(conv2b, conv2c)
pool2b = dX.pool2b(concat2)
conv3a = dX.conv3a(pool2b)
pool3a =dX.pool3a(pool2b)
conv3b = dX.conv3b(conv3a)
conv3c = dX.conv3c(pool3a)
concat3 = dX.concat3(conv3b, conv3c)
pool3b = dX.pool3b(concat3)







