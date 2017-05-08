from utils import *
import tensorflow as tf
import mnist
from model import autoencoder

NUM_GEN = 5


def main():
    with tf.Session() as sess:
        out = autoencoder.generate(sess, NUM_GEN)

    mnist.visualize_n(out)


if __name__ == '__main__':
    main()