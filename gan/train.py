import mnist
from model import *
import os
import shutil

CODE_SIZE = 128
LR = 0.01
BATCH_SIZE = 256
FINAL_STEP = 10000
CKPT_STEP = 10
CKPT_PATH = 'checkpoints/code_' + str(CODE_SIZE)
RE_TRAIN = False


def main():
    if RE_TRAIN:
        shutil.rmtree(CKPT_PATH)
    if not os.path.exists(CKPT_PATH):
        os.mkdir(CKPT_PATH)

    gen = Generator()
    disc = Discriminator()
    gd = GD(gen, disc, CKPT_PATH)

    data = mnist.load_data().train

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        gd.train(sess, data, FINAL_STEP, LR, BATCH_SIZE, writer, ckpt_step=CKPT_STEP)
        x = gen.generate(sess, CODE_SIZE, BATCH_SIZE)
        print x
        writer.close()

if __name__ == '__main__':
    main()