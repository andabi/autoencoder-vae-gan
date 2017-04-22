import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def show(input):
    img = np.reshape(np.squeeze(input), (28, 28))
    plt.imshow(img, cmap=cm.binary)
    plt.show()