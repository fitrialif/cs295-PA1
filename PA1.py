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
W = dX.weight_variable(())
b = dX.bias_variable(())
y = dX.classifier(pool3b, W, b)

cv = CV.NFoldCV(10)
sess = tf.InteractiveSession()

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(laels=y_, logits=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for i in range(20000):
    for j in range(10):
          batchx, batchy = cv.getNextBatch()
          train_step.run(feed_dict={x:batchx,y:batchy})

print ("Done")








