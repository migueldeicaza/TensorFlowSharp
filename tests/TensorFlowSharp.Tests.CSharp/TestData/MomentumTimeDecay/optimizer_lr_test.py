# This script is used to create data file (expected.txt)
# which is used to compare the output from TensorFlowSharp optimizer tests.
#
# NOTE: This script is not used to generate the expected.txt file in this case 
# because of the tf.train.MomentumOptimizer implemention difference with decay.
# The expected.txt is actually the output from the test itself.
import tensorflow as tf

# Training data
train_x =[
    3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
    7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1
]
train_y = [
    1.7, 2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
        2.827,3.465,1.65,2.904,2.42,2.94,1.3
]
n_samples = len(train_x)
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.constant(0.1), dtype=tf.float32)
b = tf.Variable(tf.constant(0.1), dtype=tf.float32)

pred = tf.add(tf.multiply(X,W), b)

global_step = tf.Variable(0, trainable=False)
learning_rate = 0.01
decay_rate = 0.5
decayed_learning_rate = learning_rate * (1. / (1. + decay_rate * tf.cast(global_step, tf.float32)))

cost = tf.divide(tf.reduce_sum(tf.pow(tf.subtract(pred, Y), 2.0)), tf.multiply(2.0, n_samples))
optimizer = tf.train.MomentumOptimizer(decayed_learning_rate, 0.9).minimize(cost, global_step=global_step, name = "MomentumOptimizer")

init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)
    for e in range(2):
        for i in range(n_samples):
            _, cost_v, W_v, b_v, pred_v, lr_v, step_v = session.run([optimizer, cost, W, b, pred, decayed_learning_rate, global_step], feed_dict = {X: train_x[i], Y: train_y[i]})
            print(f"step: {step_v:d}, loss: {cost_v:.4f}, W: {W_v:.4f}, b: {b_v:.4f}, lr: {lr_v:.8f}")
            #print("Prediction: %f == Actual: %f" % (pred_v, train_y[i]))