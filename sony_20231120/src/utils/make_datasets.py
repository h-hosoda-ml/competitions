import numpy as np
import cv2


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
    dot_padding = 15  # 目の位置を調整

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
