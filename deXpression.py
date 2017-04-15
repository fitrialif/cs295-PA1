import tensorflow as tf


def conv1(x):
    return tf.layers.conv2d(inputs=x, filters=64, kernel_size=(), strides=(), padding='valid')


def pool1(x):
    return tf.layers.max_pooling2d(x, pool_size=(), strides=(), padding='valid')


def lrn1(x):
    return tf.nn.local_response_normalization(input=x)


def conv2a(x):
    return tf.layers.conv2d(inputs=x, filters=64, kernel_size=(), strides=(), padding='valid')


def pool2a(x):
    return tf.layers.max_pooling2d(x, pool_size=(), strides=(), padding='valid')


def conv2b(x):
    return tf.layers.conv2d(inputs=x, filters=64, kernel_size=(), strides=(), padding='valid')


def conv2c(x):
    return tf.layers.conv2d(inputs=x, filters=64, kernel_size=(), strides=(), padding='valid')


def relu2b(x):
    return tf.nn.relu(x)


def relu2c(x):
    return tf.nn.relu(x)


def concat2(x,y):
    return tf.concat([x, y])


def pool2b(x):
    return tf.layers.max_pooling2d(x, pool_size=(), strides=(), padding='valid')


def conv3a(x):
    return tf.layers.conv2d(inputs=x, filters=64, kernel_size=(), strides=(), padding='valid')


def pool3a(x):
    return tf.layers.max_pooling2d(x, pool_size=(), strides=(), padding='valid')


def conv3b(x):
    return tf.layers.conv2d(inputs=x, filters=64, kernel_size=(), strides=(), padding='valid')


def conv3c(x):
    return tf.layers.conv2d(inputs=x, filters=64, kernel_size=(), strides=(), padding='valid')


def concat3(x,y):
    return tf.concat([x, y])


def pool3b(x):
    return tf.layers.max_pooling2d(x, pool_size=(), strides=(), padding='valid')


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stdev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial  = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def classifier(x, W, b):
     return tf.matmul(x,W) +b