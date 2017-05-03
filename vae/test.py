import mnist
from model import *

NUM_TEST = 3


def main():
    data = mnist.load_data().test
    visualizer = mnist.visualize
    model = VariationalAutoEncoder(X_SIZE, Z_SIZE)

    with tf.Session() as sess:
        model.test(sess, data, visualizer, NUM_TEST)


if __name__ == '__main__':
    main()