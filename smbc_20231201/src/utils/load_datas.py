import os
import sys

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from conf import DATA_PATH

# データセット名
_FILE_NAMES = ["train.csv", "test.csv"]


def load_dataset(filename: str) -> pd.DataFrame:
    """
    指定のデータセットを読み込む関数

    Attributes
    --------------
    filename: str
        読み込みたいcsvデータの名前
    """
    if not filename in _FILE_NAMES:
        raise ValueError("指定したファイル名を読み込めません")

    data_path = os.path.join(DATA_PATH, filename)
    df = pd.read_csv(data_path, index_col=0)

    return df


def load_all_dataset() -> list:
    """
    訓練データとテストデータ両方を読み込む関数
    """
    datas = []

    for file_name in _FILE_NAMES:
        data_path = os.path.join(DATA_PATH, file_name)
        df = pd.read_csv(data_path, index_col=0)
        datas.append(df)

    return datas
