import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.examples.tutorials.mnist import input_data


def load_data():
    return input_data.read_data_sets('data/mnist', one_hot=True)


def visualize(input):
    img = np.reshape(np.squeeze(input), (28, 28))
    plt.imshow(img, cmap=cm.binary)
    plt.show()