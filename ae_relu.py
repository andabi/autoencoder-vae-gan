import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import mnist_visualizer as v

BATCH_SIZE = 128
Z_SIZE = 128
H_1_SIZE = 512
H_2_SIZE = 256
LR = 0.001
TRAIN_STEPS = 500
SKIP_STEP = 1
CKPT_PATH = 'checkpoints/relu'

# class AutoEncoder(object):

mnist = input_data.read_data_sets('data/mnist', one_hot=True)

input = tf.placeholder(tf.float32, shape=(None, 784))

# encoder
w_1_encoder = tf.Variable(tf.zeros(shape=(784, H_1_SIZE)))
b_1_encoder = tf.Variable(tf.random_normal(shape=(1, H_1_SIZE)))
out_1_encoder = tf.nn.relu(tf.matmul(input, w_1_encoder) + b_1_encoder)

w_2_encoder = tf.Variable(tf.zeros(shape=(H_1_SIZE, H_2_SIZE)))
b_2_encoder = tf.Variable(tf.random_normal(shape=(1, H_2_SIZE)))
out_2_encoder = tf.nn.relu(tf.matmul(out_1_encoder, w_2_encoder) + b_2_encoder)

w_3_encoder = tf.Variable(tf.zeros(shape=(H_2_SIZE, Z_SIZE)))
b_3_encoder = tf.Variable(tf.random_normal(shape=(1, Z_SIZE)))
out_encoder = tf.nn.relu(tf.matmul(out_2_encoder, w_3_encoder) + b_3_encoder)

# decoder
w_1_decoder = tf.Variable(tf.zeros(shape=(Z_SIZE, H_2_SIZE)))
b_1_decoder = tf.Variable(tf.random_normal(shape=(1, H_2_SIZE)))
out_1_decoder = tf.nn.relu(tf.matmul(out_encoder, w_1_decoder) + b_1_decoder)

w_2_decoder = tf.Variable(tf.zeros(shape=(H_2_SIZE, H_1_SIZE)))
b_2_decoder = tf.Variable(tf.random_normal(shape=(1, H_1_SIZE)))
out_2_decoder = tf.nn.relu(tf.matmul(out_1_decoder, w_2_decoder) + b_2_decoder)

w_3_decoder = tf.Variable(tf.zeros(shape=(H_1_SIZE, 784)))
b_3_decoder = tf.Variable(tf.random_normal(shape=(1, 784)))
out_decoder = tf.nn.sigmoid(tf.matmul(out_2_decoder, w_3_decoder) + b_3_decoder)

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
loss = tf.reduce_mean(tf.square(input - out_decoder))
optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)
grad_loss = tf.gradients(loss, [w_1_encoder, w_2_encoder, w_1_decoder, w_2_decoder])
grad_out_decoder = tf.gradients(out_decoder, [w_1_encoder, w_2_encoder, w_1_decoder, w_2_decoder])

# train
saver = tf.train.Saver()

with tf.Session() as sess:
    # writer = tf.summary.FileWriter('./graph', sess.graph)
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(CKPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    num_batches = int(mnist.train.num_examples / BATCH_SIZE)
    prev_loss, d_loss = 0.0, 0.0
    for step in range(global_step.eval(), TRAIN_STEPS):

        total_loss = 0
        for _ in range(num_batches):
            input_batch, _ = mnist.train.next_batch(BATCH_SIZE)
            _, loss_batch, g_loss, g_out = sess.run([optimizer, loss, grad_loss, grad_out_decoder], feed_dict={input: input_batch})
            total_loss += loss_batch
            # print 'grad_loss={}, grad_out={}'.format([np.linalg.norm(grad) for grad in g_loss], [np.linalg.norm(grad) for grad in g_out])

        curr_loss = total_loss / num_batches
        if prev_loss:
            d_loss = (curr_loss / prev_loss - 1)
        prev_loss = curr_loss

        print 'step-{}\td_loss={:1.6f}\tavg_loss={}\tgrad_loss={}\tgrad_out={}'.format(step, d_loss, curr_loss, [np.linalg.norm(grad) for grad in g_loss], [np.linalg.norm(grad) for grad in g_out])

        sess.run(global_step.assign_add(1))

        if (step + 1) % SKIP_STEP == 0:
            saver.save(sess, CKPT_PATH + '/autoencoder', step)

    # test
    x, _ = mnist.test.next_batch(1)
    # x = np.random.uniform(size=(10, 784))
    v.show(x)
    out_1 = sess.run(out_1_encoder, feed_dict={input: x})
    out_2 = sess.run(out_2_encoder, feed_dict={out_1_encoder: out_1})
    z = sess.run(out_encoder, feed_dict={out_2_encoder: out_2})
    # out = sess.run(out_2_decoder, feed_dict={out_encoder: z})
    print(np.sum(out_1, axis=1))
    print(np.sum(out_2, axis=1))
    print(np.sum(z, axis=1))

    img = sess.run(out_decoder, feed_dict={out_encoder: z})
    print(np.sum(img))
    v.show(img)

    # W_1, b_1, W_2, b_2 = sess.run([w_1_decoder, b_1_decoder, w_2_decoder, b_2_decoder])
    # print(W_2)
    # print(b_2)
    # print(np.matmul(out, W_2))
    # z = sess.run(out_encoder, feed_dict={input: input_batch[-1:]})

    # writer.close()

# generation
# with tf.Session() as sess:
#     z = np.reshape(np.random.normal(scale=10, size=Z_SIZE), (1, Z_SIZE))
#     # z = np.ndarray(shape=(1, Z_SIZE), dtype=np.float32)
#     print(z)
#     out_1 = sess.run(out_1_decoder, feed_dict={w_1_decoder: W_1, b_1_decoder: b_1, out_encoder: z})
#     print(out_1)
#     generated_image = sess.run(out_decoder, feed_dict={out_1_decoder: out_1, w_2_decoder: W_2, b_2_decoder: b_2})
#     print(generated_image)
#     v.show(generated_image)

# def main():
#     model = AutoEncoder()
#     # model.build_graph()
#     batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
#     train_model(model, batch_gen, NUM_TRAIN_STEPS, WEIGHTS_FLD)
#
# if __name__ == '__main__':
#     main()