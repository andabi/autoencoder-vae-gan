import mnist
from model import *

CODE_SIZE = 128
CKPT_PATH = 'checkpoints/code_' + str(CODE_SIZE)


def main():
    gen = Generator(CODE_SIZE)
    disc = Discriminator()
    gd = GD(gen, disc, CKPT_PATH)

    with tf.Session() as sess:
        x = gd.generate(sess, 3)
        mnist.visualize_n(x)

if __name__ == '__main__':
    main()