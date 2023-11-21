import os
import sys

import random
import numpy as np
import cv2

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from conf import DATASET_PATH

# データの保存を行うパス
_SAVEPATH = os.path.join(DATASET_PATH, "my_datasets")


def make_dice(number: int):
    """
    サイコロを作成する関数

    Attributes
    ----------
    number: int
        作成したいサイコロの目の数
    """

    dice_size = 200  # サイコロのサイズ
    dice_surface = np.ones((200, 200), dtype=np.uint8) * 255  # サイコロの表面(白色 200x200)

    dot_size = 25  # サイコロの目の大きさ
    dot_padding = 0  # 目の位置を調整

    if number % 2 == 1:
        # サイコロの目が奇数の時、中心に描画
        cv2.circle(
            dice_surface,
            center=(dice_size // 2, dice_size // 2),
            radius=dot_size,
            color=(0, 0, 0),
            thickness=-1,
        )

    if number != 1:
        # サイコロの目が1ではない時、右上に描画
        cv2.circle(
            dice_surface,
            center=(3 * dice_size // 4 + dot_padding, dice_size // 4 - dot_padding),
            radius=dot_size,
            color=(0, 0, 0),
            thickness=-1,
        )

    if number == 4:
        # サイコロの目が4の時、左上に描画
        cv2.circle(
            dice_surface,
            center=(dice_size // 4 - dot_padding, dice_size // 4 - dot_padding),
            radius=dot_size,
            color=(0, 0, 0),
            thickness=-1,
        )

    if number in [2, 3, 4]:
        # サイコロの目が2,3,4の時、左下に描画
        cv2.circle(
            dice_surface,
            center=(dice_size // 4 - dot_padding, 3 * dice_size // 4 + dot_padding),
            radius=dot_size,
            color=(0, 0, 0),
            thickness=-1,
        )

    if number in [5, 6]:
        # サイコロの目が5,6の時、右下に描画
        cv2.circle(
            dice_surface,
            center=(3 * dice_size // 4 + dot_padding, 3 * dice_size // 4 + dot_padding),
            radius=dot_size,
            color=(0, 0, 0),
            thickness=-1,
        )

    if number == 6:
        # サイコロの目が6の時、左中心に描画
        cv2.circle(
            dice_surface,
            center=(dice_size // 4 - dot_padding, dice_size // 2),
            radius=dot_size,
            color=(0, 0, 0),
            thickness=-1,
        )

    return dice_surface


def make_data(number: int, angle: int) -> np.ndarray:
    """
    任意のダイスに対して、任意の回転を適用した配列を返す

    Attributes
    ----------
    number: int
        作成したいサイコロの目
    anlge: int
        サイコロに対して適用したい回転
    """

    # 番号に基づいてサイコロを取得
    dice_path = os.path.join(DATASET_PATH, f"dices/dice_{number}.npy")
    dice_image = np.load(dice_path)

    # 回転を適用させる
    center = tuple(np.array(dice_image.shape[::-1]) / 2)  # サイコロの中心座標
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)  # 回転行列を取得 (中心座標、回転角度、スケール)

    # 回転後の画像のサイズを計算
    rotated_width = int(
        np.ceil(dice_image.shape[1] * abs(rot_mat[0, 0]))
        + np.ceil(dice_image.shape[0] * abs(rot_mat[0, 1]))
    )
    rotated_height = int(
        np.ceil(dice_image.shape[1] * abs(rot_mat[1, 0]))
        + np.ceil(dice_image.shape[0] * abs(rot_mat[1, 1]))
    )

    # アフィン変換行列を調整して回転後の画像がトリミングされないようにする
    rot_mat[0, 2] += (rotated_width - dice_image.shape[1]) // 2
    rot_mat[1, 2] += (rotated_height - dice_image.shape[0]) // 2

    rotated_dice_image = cv2.warpAffine(
        dice_image, rot_mat, (rotated_width, rotated_height)
    )

    return rotated_dice_image


def make_dataset(image_size: tuple = (500, 500)) -> (np.ndarray, list):
    """
    学習データを生成する関数

    Attributes
    ----------
    image_size: tuple
        作成したいデータの大きさ
    """
    # 背景の生成
    background = np.zeros(image_size, dtype=np.uint8)
    # サイコロの個数をランダムに決定 1 - 3
    num = random.randint(1, 3)

    # ラベルのスケーリングを行う
    scale_size_h, scale_size_w = image_size
    # 最大試行回数 1000回
    max_attempts = 1000
    # 配置したダイスの数
    placed_dice_num = 0
    # データセットのラベルを格納
    box_label = []

    while max_attempts > 0 and placed_dice_num < num:
        # ランダムに回転とダイスの目を指定
        angle = np.random.randint(0, 360)
        dice_num = np.random.randint(1, 6)

        dice = make_data(dice_num, angle)  # 回転したダイスの生成
        dice_height, dice_width = dice.shape  # 取得したダイスの高さと幅

        # 座標をランダムに指定
        x = random.randint(0, image_size[1] - dice_width)  # x座標
        y = random.randint(0, image_size[0] - dice_height)  # y座標

        # ダイスが重なっているか確認
        overlapped = np.sum(background[y : y + dice_height, x : x + dice_width] > 0) > 0

        # 重なっていなければダイスを背景に合成
        if not overlapped:
            mask = (dice > 0).astype(np.uint8) * 255
            background[y : y + dice_height, x : x + dice_width] = cv2.bitwise_and(
                dice, dice, mask=mask
            )
            placed_dice_num += 1
            box_label.append(
                [
                    str(dice_num),
                    str((x + (dice_width / 2)) / scale_size_w),
                    str((y + (dice_height / 2)) / scale_size_h),
                    str(dice_width / scale_size_w),
                    str(dice_height / scale_size_h),
                ]
            )

        max_attempts -= 1

    return background, box_label


def generate_datasets(
    dataset_num: int, image_size: tuple = (500, 500), save_path: str = _SAVEPATH
):
    """
    指定の数だけデータセットを作成し、保存する関数

    Attributes
    -----------
    dataset_num: int
        作成したデータセットの数
    image_size: tuple
        作成したいデータセットの画像サイズを指定。保存される際には(20, 20)へ変更される。
    sava_path: str
        データセットの保存先を指定
    """
    for i in range(dataset_num):
        data, labels = make_dataset(image_size)
        # 画像データの保存
        data_name = f"image_{i}.png"  # データの名前
        data_path = os.path.join(save_path, "images/", data_name)  # 保存先

        data = cv2.resize(data, (20, 20))

        # データに対してノイズを付加
        h, w = data.shape  # データセットの大きさ
        noise_level = 75
        noise = np.random.randint(0, noise_level, (h, w))
        data = data + noise  # ノイズを付加
        data = data - data.min() / (data.max() - data.min())  # 0 - 1 スケーリング

        # 画像を保存
        cv2.imwrite(data_path, data)

        # ラベルデータの保存
        label_name = f"image_{i}.txt"  # ラベルの名前
        label_path = os.path.join(save_path, "labels/", label_name)  # 保存先
        with open(label_path, mode="w") as f:
            for label in labels:
                t = " ".join(label) + "\n"
                f.write(t)
