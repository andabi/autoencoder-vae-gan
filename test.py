from utils import *
import tensorflow as tf
import mnist
from model import autoencoder

NUM_TEST = 1


def main():
    data = mnist.load_data().test
    visualizer = mnist.visualize
    with tf.Session() as sess:
        autoencoder.test(sess, data, visualizer, NUM_TEST)


if __name__ == '__main__':
    main()