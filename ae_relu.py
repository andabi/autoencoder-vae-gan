import os

import tensorflow as tf
import numpy as np
import mnist
from utils import *

BATCH_SIZE = 256
X_SIZE = 784
Z_SIZE = 128
H_1_SIZE = 512
H_2_SIZE = 256
LR = 0.0005
FINAL_STEP = 50
CKPT_STEP = 10
CKPT_PATH = 'checkpoints/relu/'

# init
data = mnist.data
visualizer = mnist.show

input = tf.placeholder(tf.float32, shape=(BATCH_SIZE, X_SIZE))

out_1_encoder, w_1_encoder = fc_with_weight(input, H_1_SIZE)
out_2_encoder = fc(out_1_encoder, H_1_SIZE)
out_3_encoder = fc(out_2_encoder, H_2_SIZE)
out_4_encoder = fc(out_3_encoder, H_2_SIZE)
out_5_encoder, w_5_encoder = fc_with_weight(out_4_encoder, Z_SIZE)
out_encoder = fc(out_5_encoder, Z_SIZE)

# decoder
out_1_decoder, w_1_decoder = fc_with_weight(out_encoder, Z_SIZE)
out_2_decoder = fc(out_1_decoder, H_2_SIZE)
out_3_decoder = fc(out_2_decoder, H_2_SIZE)
out_4_decoder = fc(out_3_decoder, H_1_SIZE)
out_5_decoder, w_5_decoder = fc_with_weight(out_4_decoder, H_1_SIZE)
out_decoder = fc(out_5_decoder, X_SIZE)

loss = tf.reduce_mean(tf.square(input - out_decoder))
optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

grad_loss = tf.gradients(loss, [w_1_encoder, w_5_encoder, w_1_decoder, w_5_decoder])
grad_out_decoder = tf.gradients(out_decoder, [w_1_encoder, w_5_encoder, w_1_decoder, w_5_decoder])

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

# train
saver = tf.train.Saver()

with tf.Session() as sess:
    # writer = tf.summary.FileWriter('./graph', sess.graph)
    sess.run(tf.global_variables_initializer())

    ckpt = tf.train.get_checkpoint_state(os.path.dirname(CKPT_PATH + '/checkpoint'))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    num_batches = int(data.train.num_examples / BATCH_SIZE)
    prev_loss, d_loss = 0.0, 0.0
    for step in range(global_step.eval(), FINAL_STEP):

        total_loss = 0
        for _ in range(num_batches):
            input_batch, _ = data.train.next_batch(BATCH_SIZE)
            _, loss_batch, g_loss, g_out = sess.run([optimizer, loss, grad_loss, grad_out_decoder], feed_dict={input: input_batch})
            total_loss += loss_batch
            # print 'grad_loss={}, grad_out={}'.format([np.linalg.norm(grad) for grad in g_loss], [np.linalg.norm(grad) for grad in g_out])

        curr_loss = total_loss / num_batches
        if prev_loss:
            d_loss = (curr_loss / prev_loss - 1)
        prev_loss = curr_loss

        print 'step-{}\td_loss={:1.6f}\tavg_loss={}\tlast_grad_loss={}\tlast_grad_out={}'\
            .format(step, d_loss, curr_loss, [np.linalg.norm(grad) for grad in g_loss], [np.linalg.norm(grad) for grad in g_out])

        sess.run(global_step.assign_add(1))

        if (step + 1) % CKPT_STEP == 0:
            saver.save(sess, CKPT_PATH + '/autoencoder', step)

    # test
    x, _ = data.test.next_batch(1)
    # x = np.random.uniform(size=(10, 784))
    visualizer(x)
    out_1 = sess.run(out_1_encoder, feed_dict={input: x})
    out_2 = sess.run(out_2_encoder, feed_dict={out_1_encoder: out_1})
    z = sess.run(out_encoder, feed_dict={out_2_encoder: out_2})
    # out = sess.run(out_2_decoder, feed_dict={out_encoder: z})
    # print(np.sum(out_1, axis=1))
    # print(np.sum(out_2, axis=1))
    # print(np.sum(z, axis=1))

    img = sess.run(out_decoder, feed_dict={out_encoder: z})
    print(np.sum(img))
    visualizer(img)

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


# class AutoEncoder(object):
    # def main():
#     model = AutoEncoder()
#     # model.build_graph()
#     batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
#     train_model(model, batch_gen, NUM_TRAIN_STEPS, WEIGHTS_FLD)
#
# if __name__ == '__main__':
#     main()