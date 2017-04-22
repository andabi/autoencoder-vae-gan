import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_visualizer as visualizer
import numpy as np

BATCH_SIZE = 128
Z_SIZE = 256
LR = 0.01
TRAIN_STEPS = 10
ALPHA = 0

def leaky_relu(v):
    return tf.maximum(v, ALPHA * v)

mnist = input_data.read_data_sets('data/mnist', one_hot=True)

input = tf.placeholder(tf.float32, shape=(None, 784))

# encoder
w_1_encoder = tf.Variable(tf.random_normal(shape=(784, 256)))
b_1_encoder = tf.Variable(tf.random_uniform(shape=(1, 256), minval=-1, maxval=1))
out_1_encoder = leaky_relu(tf.matmul(input, w_1_encoder) + b_1_encoder)

w_2_encoder = tf.Variable(tf.random_normal(shape=(256, Z_SIZE)))
b_2_encoder = tf.Variable(tf.random_uniform(shape=(1, Z_SIZE), minval=-1, maxval=1))
out_encoder = leaky_relu(tf.matmul(out_1_encoder, w_2_encoder) + b_2_encoder)

# decoder
w_1_decoder = tf.Variable(tf.random_normal(shape=(Z_SIZE, 256)))
b_1_decoder = tf.Variable(tf.random_uniform(shape=(1, 256), minval=-1, maxval=1))
out_1_decoder = leaky_relu(tf.matmul(out_encoder, w_1_decoder) + b_1_decoder)

w_2_decoder = tf.Variable(tf.random_normal(shape=(256, 784)))
b_2_decoder = tf.Variable(tf.random_uniform(shape=(1, 784), minval=-1, maxval=1))
out_decoder = leaky_relu(tf.matmul(out_1_decoder, w_2_decoder) + b_2_decoder)

loss = tf.reduce_mean(tf.square(input - out_decoder))
optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

# train
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_batches = int(mnist.train.num_examples / BATCH_SIZE)
    for step in range(1, TRAIN_STEPS + 1):
        total_loss = 0
        for _ in range(num_batches):
            input_batch, _ = mnist.train.next_batch(BATCH_SIZE)
            _, loss_batch, z = sess.run([optimizer, loss, out_encoder], feed_dict={input: input_batch})
            total_loss += loss_batch
        print 'step{} avg loss={}'.format(step, total_loss / num_batches)
    W_1, b_1, W_2, b_2 = sess.run([w_1_decoder, b_1_decoder, w_2_decoder, b_2_decoder])
    z = sess.run(out_encoder, feed_dict={input: input_batch})

# generation
with tf.Session() as sess:
    noise = np.random.uniform(0, 0.01, 256)
    # z = np.ndarray(shape=(1, Z_SIZE), dtype=np.float32)
    z = z[:1] + noise
    generated_image = sess.run(out_decoder, feed_dict={w_1_decoder: W_1, b_1_decoder: b_1, w_2_decoder: W_2, b_2_decoder: b_2, out_encoder: z})
    visualizer.show(generated_image)