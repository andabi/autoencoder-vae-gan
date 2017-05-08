import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data


def load_data():
    return input_data.read_data_sets('data/mnist', one_hot=True)


def visualize(input):
    img = np.reshape(np.squeeze(input), (28, 28))
    plt.imshow(img, cmap="gray")
    plt.show()


def visualize_n(input):
    num = min(5, input.shape[0])
    for i in range(num):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(input[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.show()


def visualize_comp(orig, new):
    num = min(4, orig.shape[0])
    plt.figure(figsize=(6, 12))
    for i in range(num):
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(orig[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2 * i + 2)
        plt.imshow(new[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
    plt.tight_layout()
    plt.show()