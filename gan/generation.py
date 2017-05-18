import mnist
from model import *
import numpy as np
import tensorflow as tf

NUM_GEN = 10
CODE_SIZE = 50
CKPT_PATH = 'checkpoints/code_' + str(CODE_SIZE)


def main():
    gen = Generator(CODE_SIZE, NUM_GEN)
    disc = Discriminator(batch_size=NUM_GEN)
    gd = GD(gen, disc, CKPT_PATH)

    data = mnist.load_data().test

    config = tf.ConfigProto(
        device_count={'GPU': 0},
        # log_device_placement=True
    )

    with tf.Session(config=config) as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)

        x = np.random.normal(0, 1, (NUM_GEN, X_SIZE))
        d = gd.discriminate(sess, x)
        print 'random_image\t{}'.format(np.mean(d))

        x, _ = data.next_batch(NUM_GEN)
        d = gd.discriminate(sess, x)
        print 'real_image\t{}'.format(np.mean(d))

        x = gd.generate(sess, NUM_GEN)
        d = gd.discriminate(sess, x)
        print 'fake_image\t{}'.format(np.mean(d))

        x = tf.reshape(x, [NUM_GEN, 28, 28, 1])
        image_summary = tf.summary.image('generated_image', x)
        summary = sess.run(image_summary)
        writer.add_summary(summary)

        writer.close()

if __name__ == '__main__':
    main()