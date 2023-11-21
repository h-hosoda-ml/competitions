import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
sys.path.append(os.path.dirname(__file__))

from conf import DATASET_PATH
from preprocess import image_scaler

# データセット名
_FILE_NAMES = ["X_train.npy", "y_train.npy", "X_test.npy"]
_NEED_SCALES = [True, False, True]


def load_datasets(filename: str, need_scale: bool = True) -> np.ndarray:
    """
    指定のデータセットを読み出す関数
    filename: データセットのファイル名
    need_scale: 0-1にスケーリングを行うフラグ
    """
    if not filename in _FILE_NAMES:
        raise ValueError("指定したファイル名は存在しません")

    data_path = os.path.join(DATASET_PATH, filename)
    data = np.load(data_path)

    if need_scale:
        data = image_scaler(data)

    return data


def load_all_datasets(
    filenames: list = _FILE_NAMES, need_scales: bool = _NEED_SCALES
) -> tuple:
    """
    全てのデータセット読み出す関数
    filenames: 読み込みを行うファイル名のリスト
    need_scales: スケーリングを行うか各データごとに指定するリスト
    """
    return (
        load_datasets(data, need_scale)
        for data, need_scale in zip(filenames, need_scales)
    )


# 動作確認
if __name__ == "__main__":
    X_train = load_datasets("X_train.npy", True)
    print(f"shape of X_train: {X_train.shape}", end="\n\n")

    X_train, y_train, X_test = load_all_datasets()
    print(f"shape of X_train: {X_train.shape}")
    print(f"shape of y_train: {y_train.shape}")
    print(f"shape of X_test: {X_test.shape}")
