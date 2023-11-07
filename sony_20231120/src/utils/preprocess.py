import numpy as np
import matplotlib.pyplot as plt


def image_scaler(data: np.ndarray):
    data = data / 255.0
    if data.ndim == 1:
        return data.reshape(20, 20)

    else:
        return data.reshape(-1, 20, 20)
