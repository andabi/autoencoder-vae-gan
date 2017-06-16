import mnist
from model import *
import matplotlib.pyplot as plt
import numpy as np

NUM_GEN = 10
CODE_SIZE = 128
CASE = 'default'
CKPT_PATH = 'checkpoints/' + CASE


def main():
    model = VAE(code_size=CODE_SIZE, ckpt_path=CKPT_PATH)

    # with tf.Session() as sess:
    #     nx = ny = 10
    #     x_values = np.linspace(-10, 10, nx)
    #     y_values = np.linspace(-10, 10, ny)
    #     canvas = np.empty((28 * ny, 28 * nx))
    #     for i, yi in enumerate(x_values):
    #         for j, xi in enumerate(y_values):
    #             mu = np.array([xi, yi])
    #             out = model.generate(sess, mu=mu)
    #             canvas[(nx - i - 1) * 28:(nx - i) * 28, j * 28:(j + 1) * 28] = out[0].reshape(28, 28)
    #
    # plt.figure(figsize=(8, 10))
    # plt.imshow(canvas, origin="upper", cmap="gray")
    # plt.tight_layout()
    # plt.show()

    with tf.Session() as sess:
        nx = ny = 20
        x_values = np.linspace(-1, 1, nx)
        y_values = np.linspace(-1, 1, ny)
        plt.ion()
        for i, yi in enumerate(x_values):
            for j, xi in enumerate(y_values):
                mu = np.array([xi, yi])
                out = model.generate(sess, mu=mu)
                plt.imshow(out[0].reshape(28, 28), origin="upper", cmap="gray")
                plt.pause(1e-10)


if __name__ == '__main__':
    main()