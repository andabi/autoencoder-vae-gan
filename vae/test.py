import mnist
from model import *

NUM_TEST = 10
CODE_SIZE = 128
CKPT_PATH = 'checkpoints/code_' + str(CODE_SIZE)


def main():
    data = mnist.load_data().test
    model = VariationalAutoEncoder(code_size=CODE_SIZE, ckpt_path=CKPT_PATH)

    with tf.Session() as sess:
        x, _ = data.next_batch(NUM_TEST)
        x_ = model.reconstruct(sess, x)

    mnist.visualize_comp(x, x_)


if __name__ == '__main__':
    main()