import tensorflow as tf
import mnist
from model import autoencoder

NUM_TEST = 2


def main():
    data = mnist.load_data().test
    with tf.Session() as sess:
        x, _ = data.next_batch(NUM_TEST)
        mnist.visualize_n(x)
        out = autoencoder.test(sess, x)
        mnist.visualize_n(out)

if __name__ == '__main__':
    main()