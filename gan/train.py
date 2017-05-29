import mnist
from model import *
import os
import shutil

CASE = 'simple_gen_3'
CKPT_PATH = 'checkpoints/' + CASE
GRAPH_PATH = 'graphs/' + CASE
CODE_SIZE = 128
LR_DISC = 1e-4
LR_GEN = 1e-3
BATCH_SIZE = 128
FINAL_STEP = 500000
CKPT_STEP = 500
RE_TRAIN = False


def main():
    if RE_TRAIN:
        if os.path.exists(CKPT_PATH):
            shutil.rmtree(CKPT_PATH)
        if os.path.exists(GRAPH_PATH):
            shutil.rmtree(GRAPH_PATH)
    if not os.path.exists(CKPT_PATH):
        os.mkdir(CKPT_PATH)

    gen = Generator(CODE_SIZE, batch_size=BATCH_SIZE)
    disc = Discriminator(batch_size=BATCH_SIZE)
    gd = GD(gen, disc, CKPT_PATH)

    data = mnist.load_data().train

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(GRAPH_PATH, sess.graph)
        gd.train(sess, data, FINAL_STEP, LR_GEN, LR_DISC, BATCH_SIZE, writer, ckpt_step=CKPT_STEP, k_gen=3)
        writer.close()

if __name__ == '__main__':
    main()