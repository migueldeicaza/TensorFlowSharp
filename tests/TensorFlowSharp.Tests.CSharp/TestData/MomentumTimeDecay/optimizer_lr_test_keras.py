import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import numpy as np

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
model = tf.keras.Sequential()
model.add(layers.Dense(1, kernel_initializer = tf.keras.initializers.Constant(0.1, dtype=tf.float32), bias_initializer = tf.keras.initializers.Constant(0.1, dtype=tf.float32)))

def loss(y_true, y_pred):
    return tf.divide(tf.reduce_sum(tf.pow(tf.subtract(y_pred, y_true), 2.0)), tf.multiply(2.0, n_samples))

sgd = tf.keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.5)
model.compile(optimizer=sgd,
              loss=loss,
              metrics=['mse'])
sess = K.get_session()
K.set_session(sess)
for x,y in zip(train_x, train_y):
    model.fit(np.reshape(np.asarray(x, dtype=np.float32), (1,1)), np.reshape(np.asarray(y, dtype=np.float32), (1,)), epochs=1, batch_size=1,shuffle=False)
    op_y = tf.get_default_graph().get_tensor_by_name("output_1_target:0")
    print(sess.run([sgd.lr,sgd.iterations,sgd.moments, sgd.grads, sgd.print], feed_dict={
        "output_1_target:0": np.reshape(np.asarray(x, dtype=np.float32), (1,1)),
        "input_1:0": np.reshape(np.asarray(y, dtype=np.float32), (-1,1))}))