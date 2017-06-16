import mnist
from model import *

NUM_TEST = 10
CODE_SIZE = 128
CASE = 'default'
CKPT_PATH = 'checkpoints/' + CASE


def main():
    data = mnist.load_data().test
    model = VAE(code_size=CODE_SIZE, ckpt_path=CKPT_PATH)

    with tf.Session() as sess:
        x, _ = data.next_batch(NUM_TEST)
        x_ = model.reconstruct(sess, x)

    mnist.visualize_comp(x, x_)


if __name__ == '__main__':
    main()