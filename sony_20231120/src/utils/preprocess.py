import numpy as np
import matplotlib.pyplot as plt


def image_scaler(data: np.ndarray):
    data = data / 255.0
    data_shaped = data.reshape(20, 20)

    return data_shaped
