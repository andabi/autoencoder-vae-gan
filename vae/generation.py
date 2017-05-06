import mnist
from model import *

NUM_GEN = 10


def main():
    visualizer = mnist.visualize
    model = VariationalAutoEncoder()

    with tf.Session() as sess:
        model.generate(sess, visualizer, NUM_GEN)

if __name__ == '__main__':
    main()