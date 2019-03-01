# This script is used to create data file (expected.txt)
# which is used to compare the output from TensorFlowSharp optimizer tests.
#
# NOTE: This script is not used to generate the expected.txt file in this case 
# because of the tf.train.MomentumOptimizer implemention difference with decay.
# The expected.txt is actually the output from the test itself.

import tensorflow as tf
from keras.utils.np_utils import to_categorical
import math
import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train.reshape((-1,784))
x_test = x_test.reshape((-1,784))

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

n_samples = len(x_train)
learning_rate = 0.1
X = tf.placeholder(tf.float32, shape=[None, 784])
Y = tf.placeholder(tf.float32, shape=[None, 10])

tf.set_random_seed(1)
initB = 4 * math.sqrt(6) / math.sqrt(784 + 500)
W1 = tf.Variable(tf.random_uniform([x_train.shape[1], 500], minval=-initB, maxval=initB)) 
b1 = tf.Variable(tf.constant(0., shape=[500], dtype=tf.float32))
layer1 = tf.nn.sigmoid(tf.add(tf.matmul(X,W1), b1))

initB = 4 * math.sqrt(6) / math.sqrt(500 + 100)
W2 = tf.Variable(tf.random_uniform([500, 100], minval=-initB, maxval=initB))
b2 = tf.Variable(tf.constant(0., shape=[100], dtype=tf.float32))
layer2 = tf.nn.sigmoid(tf.add(tf.matmul(layer1,W2), b2))

initB = 4 * math.sqrt(6) / math.sqrt(100 + 10)
W3 = tf.Variable(tf.random_uniform([100, 10], minval=-initB, maxval=initB))
b3 = tf.Variable(tf.constant(0., shape=[10], dtype=tf.float32))
layer3 = tf.add(tf.matmul(layer2,W3), b3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=layer3))
optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9).minimize(cost, name = "SGDOptimizer")

prediction = tf.nn.softmax(layer3, name = "Prediction")
accuracy = tf.reduce_mean( tf.cast(tf.equal( tf.argmax(prediction,1), tf.argmax(Y, 1)), tf.float32), name = "Accuracy")

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    batch_size =100
    total_batch = int(x_train.shape[0] / batch_size)
    for epoch in range(5):
        avg_loss = 0
        avg_acc = 0
        for batch_idx in range(0, x_train.shape[0], batch_size):
            X_batch = x_train[batch_idx:batch_idx+batch_size]
            Y_batch = y_train[batch_idx:batch_idx+batch_size]
            _, loss_val, acc = sess.run([optimizer, cost, accuracy], feed_dict={X: X_batch, Y: Y_batch})
            avg_loss += loss_val / total_batch
            avg_acc += acc / total_batch
        print('Epoch: ', '%04d' % (epoch+1), 'cost (cross-entropy) = %.4f , acc = %.4f' % (avg_loss, avg_acc))