import os

from utils import fc, fc_with_variables
import tensorflow as tf


class AutoEncoder(object):
    def __init__(self, encoder, decoder, input, input_size, code_size, ckpt_path='checkpoints/'):
        self.input = input
        self.encoder = encoder
        self.decoder = decoder
        self.input_size = input_size
        self.code_size = code_size
        self.ckpt_path = ckpt_path
        self.saver = tf.train.Saver()

    def _load(self, sess):
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(self.ckpt_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)

    # writer = tf.summary.FileWriter('./graph', sess.graph)
    # writer.close()

    def train(self, sess, data, final_step, lr, batch_size, ckpt_step=1):

        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        loss = tf.reduce_mean(tf.square(self.input - self.decoder))
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        self._load(sess)
        # grad_loss = tf.gradients(loss, [w_1_encoder, w_5_encoder, w_1_decoder, w_5_decoder])
        # grad_out_decoder = tf.gradients(out_decoder, [w_1_encoder, w_5_encoder, w_1_decoder, w_5_decoder])

        num_batches = int(data.num_examples / batch_size)
        prev_loss, d_loss = 0.0, 0.0
        for step in range(global_step.eval(), final_step):

            total_loss = 0
            for _ in range(num_batches):
                input_batch, _ = data.next_batch(batch_size)
                # _, loss_batch, g_loss, g_out = sess.run([optimizer, loss, grad_loss, grad_out_decoder], feed_dict={input: input_batch})
                _, loss_batch = sess.run([optimizer, loss], feed_dict={self.input: input_batch})
                total_loss += loss_batch
                # print 'grad_loss={}, grad_out={}'.format([np.linalg.norm(grad) for grad in g_loss], [np.linalg.norm(grad) for grad in g_out])

            curr_loss = total_loss / num_batches
            if prev_loss:
                d_loss = (curr_loss / prev_loss - 1)
            prev_loss = curr_loss

            print 'step-{}\td_loss={:2.2f}%\tavg_loss={}'.format(step, d_loss * 100, curr_loss)
            # .format(step, d_loss * 100, curr_loss, [np.linalg.norm(grad) for grad in g_loss], [np.linalg.norm(grad) for grad in g_out])

            sess.run(global_step.assign_add(1))

            if (step + 1) % ckpt_step == 0:
                self.saver.save(sess, self.ckpt_path + '/autoencoder', step)

    def test(self, sess, data, visualizer, num=2):
        x, _ = data.next_batch(num)
        for o in x:
            visualizer(o)

        self._load(sess)
        z = sess.run(self.encoder, feed_dict={self.input: x})
        out = sess.run(self.decoder, feed_dict={self.encoder: z})
        for o in out:
            visualizer(o)

    def generate(self, sess, z, visualizer):
        self._load(sess)
        out = sess.run(self.decoder, feed_dict={self.encoder: z})
        for o in out:
            visualizer(o)

X_SIZE = 784
Z_SIZE = 128
H_1_SIZE = 512
H_2_SIZE = 256


def _encoder(input, code_size):
    out_1_encoder, w_1_encoder = fc_with_variables(input, H_1_SIZE)
    # out_2_encoder = fc(out_1_encoder, H_1_SIZE)
    # out_3_encoder = fc(out_2_encoder, H_2_SIZE)
    # out_4_encoder = fc(out_3_encoder, H_2_SIZE)
    out_5_encoder, w_5_encoder = fc_with_variables(out_1_encoder, code_size)
    out_encoder = fc(out_5_encoder, code_size)
    return out_encoder


def _decoder(code, code_size, out_size):
    out_1_decoder, w_1_decoder = fc_with_variables(code, code_size)
    # out_2_decoder = fc(out_1_decoder, H_2_SIZE)
    # out_3_decoder = fc(out_2_decoder, H_2_SIZE)
    # out_4_decoder = fc(out_3_decoder, H_1_SIZE)
    out_5_decoder, w_5_decoder = fc_with_variables(out_1_decoder, H_1_SIZE)
    out_decoder = fc(out_5_decoder, out_size, tf.nn.sigmoid)
    return out_decoder


input = tf.placeholder(tf.float32, shape=(None, X_SIZE))
encoder = _encoder(input, Z_SIZE)
decoder = _decoder(encoder, Z_SIZE, X_SIZE)
autoencoder = AutoEncoder(encoder, decoder, input, X_SIZE, Z_SIZE)