import sys
import os

import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from .preprocess import image_scaler


def show_image(data: np.ndarray, label: np.uint8 = None):
    """
    二次元画像をグレースケールで描画する関数

    Attributes
    -------------
    data: np.ndarray
        描画を行いたい二次元配列

    label: np.uint8
        タイトルにラベルをつけたい場合に指定
    """
    plt.imshow(data, cmap="gray")
    plt.xticks([])
    plt.yticks([])
    if label is not None:
        plt.title("label: " + str(label))
    plt.show()
