import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os
import matplotlib.image as mpimg

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def usps_test_data():
    x_usps = np.ndarray(shape=(1,784),dtype=float)
    y_usps = np.ndarray(shape=(1,10),dtype=float)

    for i in range(0, 10):
        y_t = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        y_t[i] = 1
        for image_file_name in os.listdir('USPS_norm_data/Numerals/' + str(i) + '/'):
            if image_file_name.endswith(".png"):
                im = mpimg.imread('USPS_norm_data/Numerals/' + str(i) + '/' + image_file_name)
                x_t = np.reshape(im, [1, 784])
                x_t = np.absolute(np.subtract(x_t,1))
                x_usps = np.append(x_usps, x_t, axis=0)
                y_t = np.reshape(y_t, [1, 10])
                y_usps = np.append(y_usps, y_t, axis=0)

    return x_usps, y_usps

x_u, y_u = usps_test_data()


# Logistic Regression

lr = [0.001, 0.003, 0.005, 0.08, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5]
epochs = [1, 3, 5, 8]

for i in range(0, len(lr)):
    for j in range(0, len(epochs)):
        x = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))

        y = tf.nn.softmax(tf.matmul(x, W) + b)

        # To implement cross-entropy we need to first add a new placeholder
        y_ = tf.placeholder(tf.float32, [None, 10])

        # cross-entropy function
        cross_entropy = tf.reduce_mean( -tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

        # apply your choice of optimization algorithm to modify the variables and reduce the loss
        train_step = tf.train.GradientDescentOptimizer(lr[i]).minimize(cross_entropy)

        # launch the model in an InteractiveSession
        sess = tf.InteractiveSession()

        # creating an operation to initialize the variables we created
        tf.global_variables_initializer().run()

        for _ in range(epochs[j]):
            # running the training step 1000 times
            for _ in range(1000):
                batch_xs, batch_ys = mnist.train.next_batch(100)
                sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # check if our prediction matches the truth
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

        # casting to floating point numbers and then take the mean
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # accuracy on our test data
        print('Logistic Accuracy on test MNIST data lr '+str(lr[i])+' epoch '+str(epochs[j])+' : ', sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

        print('Logistic Accuracy on USPS data lr '+str(lr[i])+' epoch '+str(epochs[j])+' : ', sess.run(accuracy, feed_dict={x: x_u, y_:y_u}))


# Single layer Neural network

lr = [0.003, 0.01, 0.1, 1]
epochs = [10, 20, 30]
nodes = [256, 512, 1024, 1568]

for i in range(0, len(lr)):
    for j in range(0, len(epochs)):
        for k in range(0, len(nodes)):
            x = tf.placeholder(tf.float32, [None, 784])
            W1 = tf.Variable(tf.random_normal([784, nodes[k]]))
            W2 = tf.Variable(tf.random_normal([nodes[k], 10]))
            b1 = tf.Variable(tf.random_normal([nodes[k]]))
            b2 = tf.Variable(tf.random_normal([10]))

            L1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
            y = tf.add(tf.matmul(L1, W2), b2)

            # To implement cross-entropy we need to first add a new placeholder
            y_ = tf.placeholder(tf.float32, [None, 10])

            learning_rate = lr[i]
            # apply your choice of optimization algorithm to modify the variables and reduce the loss
            # define cost & optimizer
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

            # launch the model in an InteractiveSession
            sess = tf.InteractiveSession()

            # creating an operation to initialize the variables we created
            tf.global_variables_initializer().run()

            training_epochs = epochs[j]
            batch_size = 100
            display_step = 1
            # running the training step in various epoch times for training
            for epoch in range(training_epochs):
                avg_cost = 0
                total_batch = int(mnist.train.num_examples / batch_size)
                for _ in range(batch_size):
                    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                    sess.run(optimizer, feed_dict={x: batch_xs, y_: batch_ys})
                    avg_cost += sess.run(cost, feed_dict={x: batch_xs, y_: batch_ys}) / total_batch

                if epoch % display_step == 0:
                    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
            # check if our prediction matches the truth
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

            # casting to floating point numbers and then take the mean
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            # accuracy on our test data
            print('Single NN Accuracy on test MNIST data nodes '+str(nodes[k])+'lr '+str(lr[i])+' epoch '+str(epochs[j])+' : ', sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
            print('Single NN Accuracy on test USPS data nodes '+str(nodes[k])+'lr '+str(lr[i])+' epoch '+str(epochs[j])+' : ', sess.run(accuracy, feed_dict={x: x_u, y_: y_u}))

# CNN

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# First CN Layer

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second CN Layer

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely Connected Layer

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

#Dropout

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

#Readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

#Train and Evaluate the model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print('step %d, training accuracy: %g' % (i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})
    print('CNN MNIST test accuracy: %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
    print('CNN USPS accuracy: %g' % accuracy.eval(feed_dict={x: x_u, y_: y_u, keep_prob: 1.0}))
