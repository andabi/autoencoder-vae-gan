from utils import *
import tensorflow as tf
import mnist
from model import autoencoder
import numpy as np

NUM_GEN = 5


def main():
    visualizer = mnist.visualize
    latent = np.random.uniform(size=(NUM_GEN, autoencoder.code_size))
    with tf.Session() as sess:
        autoencoder.generate(sess, latent, visualizer)


if __name__ == '__main__':
    main()