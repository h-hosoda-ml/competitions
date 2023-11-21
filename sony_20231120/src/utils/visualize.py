import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from .preprocess import image_scaler


def show_image(data: np.ndarray, label: np.uint8 = None):
    """
    data: 20x20のndarray
    label: 正解ラベル
    """
    plt.imshow(data, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    if label is not None:
        plt.title("label: " + str(label))
    plt.show()
