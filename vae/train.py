import mnist
from model import *
import os
import shutil

CODE_SIZE = 128
LR = 0.005
BATCH_SIZE = 256
FINAL_STEP = 10000
CKPT_STEP = 100
CKPT_PATH = 'checkpoints/code_' + str(CODE_SIZE)
RE_TRAIN = False


def main():
    if RE_TRAIN:
        shutil.rmtree(CKPT_PATH)
    if not os.path.exists(CKPT_PATH):
        os.mkdir(CKPT_PATH)

    model = VariationalAutoEncoder(code_size=CODE_SIZE, ckpt_path=CKPT_PATH)
    data = mnist.load_data().train

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./graphs', sess.graph)
        model.train(sess, data, FINAL_STEP, LR, BATCH_SIZE, writer, CKPT_STEP)
        writer.close()

if __name__ == '__main__':
    main()