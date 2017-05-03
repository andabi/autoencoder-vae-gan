import mnist
from model import *

NUM_GEN = 5


def main():
    visualizer = mnist.visualize
    model = VariationalAutoEncoder(X_SIZE, Z_SIZE)

    with tf.Session() as sess:
        model.generate(sess, visualizer, NUM_GEN)


if __name__ == '__main__':
    main()