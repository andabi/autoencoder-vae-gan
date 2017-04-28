from utils import *
import tensorflow as tf
import mnist
from model import autoencoder

LR = 0.001
FINAL_STEP = 300
BATCH_SIZE = 64
CKPT_STEP = 10

def main():
    data = mnist.data.train
    with tf.Session() as sess:
        autoencoder.train(sess, data, FINAL_STEP, LR, BATCH_SIZE, CKPT_STEP)


if __name__ == '__main__':
    main()