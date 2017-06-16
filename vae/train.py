import mnist
from model import *
import os
import shutil

CASE = 'default'
CKPT_PATH = 'checkpoints/' + CASE
GRAPH_PATH = 'graphs/' + CASE
CODE_SIZE = 128
LR = 0.001
BATCH_SIZE = 256
FINAL_STEP = 10000
CKPT_STEP = 1000
RE_TRAIN = False


def main():
    if RE_TRAIN:
        if os.path.exists(CKPT_PATH):
            shutil.rmtree(CKPT_PATH)
        if os.path.exists(GRAPH_PATH):
            shutil.rmtree(GRAPH_PATH)
    if not os.path.exists(CKPT_PATH):
        os.mkdir(CKPT_PATH)

    model = VAE(code_size=CODE_SIZE, ckpt_path=CKPT_PATH)
    data = mnist.load_data().train

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(GRAPH_PATH, sess.graph)
        model.train(sess, data, FINAL_STEP, LR, BATCH_SIZE, writer, CKPT_STEP)
        writer.close()

if __name__ == '__main__':
    main()