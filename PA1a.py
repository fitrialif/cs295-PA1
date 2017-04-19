import crossvalidation as cv
import tensorflow as tf
import time
import numpy as np

############################
# Create the network/ graph according to paper.
# print out variable names so we  an check layer output dimensions vs paper.
############################
with tf.name_scope("Inputs"):
    x = tf.placeholder(tf.uint8, shape=(1, 224, 224, 1), name="x")
    print "x:{}".format(x)
    y_ = tf.placeholder(tf.uint8, shape=(1, 7), name="y_")
    print "y_:{}".format(y_)


conv1 = tf.layers.conv2d(inputs=tf.cast(x, dtype=tf.float32), filters=64, kernel_size=(7, 7), strides=(2, 2),
                         padding='same',  name='conv1')
print "conv1:{}".format(conv1)

pool1 = tf.layers.max_pooling2d(conv1, pool_size=(3, 3), strides=(2, 2), padding='same', name='pool1')
print "pool1:{}".format(pool1)

lrn1 = tf.nn.local_response_normalization(input=pool1, name='lrn1')
print "lrn1:{}".format(lrn1)

with tf.name_scope("Feat_Ex_1"):
    conv2a = tf.layers.conv2d(inputs=lrn1, filters=96, kernel_size=(1, 1), strides=(1, 1),
                              padding='same', name='conv2a')
    print "conv2a:{}".format(conv2a)
    conv2b = tf.layers.conv2d(inputs=conv2a, filters=208, kernel_size=(3, 3), strides=(1, 1),
                              padding='same', name='conv2b')
    print "conv2b:{}".format(conv2b)
    pool2a = tf.layers.max_pooling2d(lrn1, pool_size=(3, 3), strides=(1, 1),  padding='same', name='pool2a')
    print "pool2a:{}".format(pool2a)
    conv2c = tf.layers.conv2d(inputs=pool2a, filters=64, kernel_size=(1, 1), strides=(1, 1),
                              padding='same', name='conv2c')
    print "conv2c:{}".format(conv2c)
    concat2 = tf.concat([conv2b, conv2c], axis=3, name='concat2')
    print "concat2:{}".format(concat2)
    pool2b = tf.layers.max_pooling2d(inputs=concat2, pool_size=(3, 3), strides=(2, 2), padding='same', name='pool2b')
    print "pool2b:{}".format(pool2b)

with tf.name_scope("Feat_Ex_2"):
    conv3a = tf.layers.conv2d(inputs=pool2b, filters=96, kernel_size=(1, 1), strides=(1, 1),
                              padding='same', name='conv3a')
    print "conv3a:{}".format(conv3a)
    pool3a = tf.layers.max_pooling2d(inputs=pool2b, pool_size=(3, 3), strides=(1, 1), padding='same', name='pool3a')
    print "pool3a:{}".format(pool3a)
    conv3b = tf.layers.conv2d(inputs=conv3a, filters=208, kernel_size=(3, 3), strides=(1, 1),
                              padding='same', name='conv3b')
    print "conv3b:{}".format(conv3b)
    conv3c = tf.layers.conv2d(inputs=pool3a, filters=64, kernel_size=(1, 1), strides=(1, 1),
                              padding='same', name='conv3c')
    print "conv3c:{}".format(conv3c)
    concat3 = tf.concat([conv3b, conv3c], axis=3, name='concat3')
    print "concat3:{}".format(concat3)
    pool3b = tf.layers.max_pooling2d(inputs=concat3, pool_size=(3, 3), strides=(2, 2), padding='same', name='pool3b')
    print "pool3b:{}".format(pool3b)

# Dropout Layer
with tf.name_scope("Dropout"):
    keep_prob = tf.placeholder(tf.float32, name="Keep_Prob")
    h_drop = tf.nn.dropout(pool3b, keep_prob, name='Dropout')

with tf.name_scope("Classifier"):
    def weight_variable(shape):
        initial = tf.truncated_normal(shape=shape,  stddev=0.1)
        return tf.Variable(initial)


    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    # Reshape incoming multidimensional tensor to be flat,
    # so we can create a fully connected layer with just 1 dimension
    reshaped = tf.reshape(pool3b, shape=(1, 272*14*14), name='reshaped')
    print "reshaped:{}".format(reshaped)
    y = tf.layers.dense(inputs=reshaped, units=7, name='y')
    print "y:{}".format(y)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
tf.summary.scalar("cross_entropy", cross_entropy)

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
prediction = tf.argmax(y,1)
#correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# ######################################################## Running starts here#####
# We use 10-fold Cross Validation
f10cv = cv.NFoldCV(10)
print "Starting: {}".format(time.strftime("%H:%M:%S"))
sess = tf.InteractiveSession()
kp = 0.50
print "Keep Prob={}".format(kp)
sess.run(tf.global_variables_initializer())
merged_summary = tf.summary.merge_all()

writer = tf.summary.FileWriter("./temp/PA1a/norelu/do{}".format(kp*100))
writer.add_graph(sess.graph)

# run for 10 epochs
for k in range(1):
    # reset average accuracy for this epoch.
    # An epoch is one run over the entire dataset.
    ave_acc = 0
    # Since we're using 10 fold Cross Validation, one epoch will be composed of 10 training+validating runs
    for j in range(10):
        # get the training and validation data and labels
        vset, vlabels, tset, tlabels = f10cv.getBatch(j)
        print "@ {}: Run#:{}, Fold#:{}".format(time.strftime("%H:%M:%S"), k, j)
        # training
        for i in range(len(tset)):
            # for each data point in the trainingset, feed into network
            train_step.run(feed_dict={x: tset[i], y_: tlabels[i], keep_prob: kp })
            s = sess.run(merged_summary, feed_dict={x: tset[i], y_: tlabels[i], keep_prob:kp})
            writer.add_summary(s, j*10+i)
        acc = 0
        # test
        conflog = open("conf{}_do{}.log".format(j,kp*100), "w")
        for i in range(len(vset)):
            # for each data point in the validation set, feed into network, check accuracy
            # accuracy will either be "1.0" or "0.0" depending if the label matches the output
            pred = prediction.eval(feed_dict={x: vset[i], y_: vlabels[i], keep_prob:1.00})

            p = int(pred[0])
            label = int(np.argmax(vlabels[i]))

            if p == label:
                acc +=1
            conflog.write("{}\t{}\t{}\n".format(p, label, "correct" if p == label else "incorrect"))
        acc = acc*1.0/len(vset)
        # get and print the average across the validation set
        print ("acc:{:00.2f}".format(acc*100))
        conflog.write("acc:{:00.2f}\n".format(acc*100))
        conflog.close()

        # accumulate the accuracies to measure the average for this epoch
        ave_acc += acc
        tf.summary.scalar("accuracy", acc)

    # compute average accuracy for this epoch
    ave_acc = ave_acc/10
    print ("ave acc:{:00.3f}".format(ave_acc*100))
print ("Done")
