import deXpression as dX
import crossvalidation as CV

import tensorflow as tf

x = tf.placeholder(tf.int8, shape=(244, 244))
y_ = tf.placeholder(tf.int8, shape=[7,None])
conv1 =tf.layers.conv2d(inputs=x, filters=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
pool1 = tf.layers.max_pooling2d(conv1, pool_size=(3, 3), strides=(2, 2), padding='valid')
lrn1 = tf.nn.local_response_normalization(input=pool1)
conv2a = tf.layers.conv2d(inputs=lrn1, filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid')
pool2a = tf.layers.max_pooling2d(lrn1, pool_size=(3, 3), strides=(1, 1), padding='valid')
conv2b = tf.layers.conv2d(inputs=conv2a, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid')
conv2c = tf.layers.conv2d(inputs=pool2a, filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid')
concat2 = tf.concat([conv2b, conv2c])
pool2b = tf.layers.max_pooling2d(inputs=concat2, pool_size=(3, 3), strides=(2, 2), padding='valid')
conv3a = tf.layers.conv2d(inputs=pool2b, filters=64, kernel_size=(1,1), strides=(1,1), padding='valid')
pool3a = tf.layers.max_pooling2d(inputs=pool2b, pool_size=(1, 1), strides=(3, 3), padding='valid')
conv3b = tf.layers.conv2d(inputs=conv3a, filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid')
conv3c = tf.layers.conv2d(inputs=pool3a, filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid')
concat3 = tf.concat([conv3b, conv3c])
pool3b = tf.layers.max_pooling2d(inputs=concat3, pool_size=(3, 3), strides=(2, 2), padding='valid')

W = dX.weight_variable((None, 7))
b = dX.bias_variable((None, 7))
y = dX.classifier(pool3b, W, b)

cv = CV.NFoldCV(10)
sess = tf.InteractiveSession()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(laels=y_, logits=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for k in range(10):
    for j in range(10):
        vset, tset, vlabels, tlabels = cv.getBatch(j)
        print "Run#{}, Fold#{}".format(k,j)
        for i in range(len(tset)):
            train_step.run(feed_dict={x:tset[i],y:tlabels[i]})
            print conv1
            print pool1
        print "Validation Accuracy:{}".format(accuracy.eval(feed_dict={x:vset,y:vlabels}))

print ("Done")








