import os

import tensorflow as tf
import numpy as np
import mnist

BATCH_SIZE = 64
Z_SIZE = 128
H_1_SIZE = 512
H_2_SIZE = 256
LR = 0.01
TRAIN_STEPS = 50
SKIP_STEP = 5
CKPT_PATH = 'checkpoints/sigmoid'
# class AutoEncoder(object):

# data = input_data.read_data_sets('data/mnist', one_hot=True)

input = tf.placeholder(tf.float32, shape=(None, 784))

# encoder
w_1_encoder = tf.Variable(tf.zeros(shape=(784, Z_SIZE)))
b_1_encoder = tf.Variable(tf.random_uniform(shape=(1, Z_SIZE)))
out_encoder = tf.nn.sigmoid(tf.matmul(input, w_1_encoder) + b_1_encoder)

# decoder
w_1_decoder = tf.Variable(tf.zeros(shape=(Z_SIZE, 784)))
b_1_decoder = tf.Variable(tf.random_uniform(shape=(1, 784)))
out_decoder = tf.nn.sigmoid(tf.matmul(out_encoder, w_1_decoder) + b_1_decoder)

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
loss = tf.reduce_mean(tf.square(input - out_decoder))
optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss, global_step=global_step)
grad_loss = tf.gradients(loss, [w_1_encoder, w_1_decoder])
grad_out_decoder = tf.gradients(out_decoder, [w_1_encoder, w_1_decoder])

# train
saver = tf.train.Saver()

with tf.Session() as sess:
    # writer = tf.summary.FileWriter('./graph', sess.graph)
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(CKPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    initial_step = global_step.eval() / BATCH_SIZE
    print initial_step
    if initial_step < TRAIN_STEPS:
        num_batches = int(mnist.data.train.num_examples / BATCH_SIZE)

        for step in range(initial_step, TRAIN_STEPS):
            total_loss = 0
            for _ in range(num_batches):
                input_batch, _ = mnist.data.train.next_batch(BATCH_SIZE)
                _, loss_batch, g_loss, g_out = sess.run([optimizer, loss, grad_loss, grad_out_decoder], feed_dict={input: input_batch})
                # z = sess.run(out_encoder, feed_dict={input: input_batch})
                # print(z)
                total_loss += loss_batch
                # print 'grad_loss={}, grad_out={}'.format([np.linalg.norm(grad) for grad in g_loss], [np.linalg.norm(grad) for grad in g_out])
            print 'step{}, avg_loss={}, grad_loss={}, grad_out={}'.format(step, total_loss / num_batches, [np.linalg.norm(grad) for grad in g_loss], [np.linalg.norm(grad) for grad in g_out])

            if (step + 1) % SKIP_STEP == 0:
                saver.save(sess, CKPT_PATH + '/autoencoder', step)

    # test
    x, _ = mnist.data.test.next_batch(1)
    # x = np.random.uniform(size=(10, 784))
    mnist.show(x)
    # out_1 = sess.run(out_1_encoder, feed_dict={input: x})
    # out_2 = sess.run(out_2_encoder, feed_dict={out_1_encoder: out_1})
    z = sess.run(out_encoder, feed_dict={input: x})
    # out = sess.run(out_2_decoder, feed_dict={out_encoder: z})
    # print(np.sum(out_1, axis=1))
    # print(np.sum(out_2, axis=1))
    print(np.sum(z, axis=1))

    imgs = sess.run(out_decoder, feed_dict={out_encoder: z})
    for img in imgs:
        mnist.show(img)

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