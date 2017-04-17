import crossvalidation as CV
import tensorflow as tf
import sys

x = tf.placeholder(tf.uint8, shape=(1, 224, 224, 1))
print "x:{}".format(x)
y_ = tf.placeholder(tf.uint8, shape=[1, 7])
print "y_:{}".format(y_)

conv1 = tf.layers.conv2d(inputs=tf.cast(x,dtype=tf.float32), filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', name='conv1')
print "conv1:{}".format(conv1)

pool1 = tf.layers.max_pooling2d(conv1, pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')
print "pool1:{}".format(pool1)

lrn1 = tf.nn.local_response_normalization(input=pool1, name='lrn1')
print "lrn1:{}".format(lrn1)

conv2a = tf.layers.conv2d(inputs=lrn1, filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', name='conv2a')
print "conv2a:{}".format(conv2a)


conv2b = tf.layers.conv2d(inputs=conv2a, filters=208, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv2b')
print "conv2b:{}".format(conv2b)

pool2a = tf.layers.max_pooling2d(lrn1, pool_size=(3, 3), strides=(1, 1), padding='same', name='pool2a')
print "pool2a:{}".format(pool2a)

conv2c = tf.layers.conv2d(inputs=pool2a, filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', name='conv2c')
print "conv2c:{}".format(conv2c)

concat2 = tf.concat([conv2b, conv2c], axis=3, name='concat2')
print "concat2:{}".format(concat2)

pool2b = tf.layers.max_pooling2d(inputs=concat2, pool_size=(3, 3), strides=(2, 2), padding='same', name='pool2b')
print "pool2b:{}".format(pool2b)

conv3a = tf.layers.conv2d(inputs=pool2b, filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', name='conv3a')
print "conv3a:{}".format(conv3a)

pool3a = tf.layers.max_pooling2d(inputs=pool2b, pool_size=(3, 3), strides=(1, 1), padding='same', name='pool3a')
print "pool3a:{}".format(pool3a)

conv3b = tf.layers.conv2d(inputs=conv3a, filters=208, kernel_size=(3, 3), strides=(1, 1), padding='same', name='conv3b')
print "conv3b:{}".format(conv3b)

conv3c = tf.layers.conv2d(inputs=pool3a, filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', name='conv3c')
print "conv3c:{}".format(conv3c)

concat3 = tf.concat([conv3b, conv3c], axis=3, name='concat3')
print "concat3:{}".format(concat3)

pool3b = tf.layers.max_pooling2d(inputs=concat3, pool_size=(3, 3), strides=(2, 2), padding='same', name='pool3b')
print "pool3b:{}".format(pool3b)


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape,  stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
reshaped = tf.reshape(pool3b, shape=(1, 272*14*14), name='reshaped')
print "reshaped:{}".format(reshaped)
y = tf.layers.dense(inputs=reshaped, units=7, name='ouput')
print "y:{}".format(y)

cv = CV.NFoldCV(10)
sess = tf.InteractiveSession()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for k in range(10):
    for j in range(10):
        vset, tset, vlabels, tlabels = cv.getBatch(j)
        print "Run#{}, Fold#{}".format(k, j)
        for i in range(len(tset)):
            train_step.run(feed_dict={x: tset[i], y: tlabels[i]})
            print conv1
            print pool1
        print "Validation Accuracy:{}".format(accuracy.eval(feed_dict={x:vset, y: vlabels}))

print ("Done")








