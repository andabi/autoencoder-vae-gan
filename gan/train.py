import mnist
from model import *
import os
import shutil

CODE_SIZE = 51
LR_DISC = 1e-4
LR_GEN = LR_DISC * 10
BATCH_SIZE = 100
FINAL_STEP = 500000
CKPT_STEP = 500
CKPT_PATH = 'checkpoints/code_' + str(CODE_SIZE)
RE_TRAIN = True


def main():
    if RE_TRAIN and os.path.exists(CKPT_PATH):
        shutil.rmtree(CKPT_PATH)
    if not os.path.exists(CKPT_PATH):
        os.mkdir(CKPT_PATH)

    gen = Generator(CODE_SIZE, batch_size=BATCH_SIZE)
    disc = Discriminator(batch_size=BATCH_SIZE)
    gd = GD(gen, disc, CKPT_PATH)

    data = mnist.load_data().train

    with tf.Session() as sess:

        writer = tf.summary.FileWriter('./graphs', sess.graph)
        gd.train(sess, data, FINAL_STEP, LR_GEN, LR_DISC, BATCH_SIZE, writer, ckpt_step=CKPT_STEP, k_disc=1)
        writer.close()

if __name__ == '__main__':
    main()