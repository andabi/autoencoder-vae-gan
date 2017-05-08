import mnist
from model import *

NUM_GEN = 10
CODE_SIZE = 128
CKPT_PATH = 'checkpoints/code_' + str(CODE_SIZE)


def main():
    model = VariationalAutoEncoder(code_size=CODE_SIZE, ckpt_path=CKPT_PATH)

    with tf.Session() as sess:
        out = model.generate(sess, NUM_GEN)

    mnist.visualize_n(out)

if __name__ == '__main__':
    main()