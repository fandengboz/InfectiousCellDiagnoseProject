# -*- coding: UTF-8 -*-
"""
*@ description: 处理图像的函数方法
*@ name:	ProcessImage.py
*@ author: dengbozfan
*@ time:	2025/04/16 20:25
"""

from .. import np,math
from typing import Optional, Tuple, Union

def normalize(ori_image):
    """
    description:  对于非0区域进行标准化, 并规范数据范围[t~b]
        Args:
            ori_image: 对原始图像进行归一化
        return:
            处理好的数据 slice
    """

    # 删除最低分和最高分,是数据更加"平均"
    b = np.percentile(ori_image,99)
    t = np.percentile(ori_image,1)
    slice = np.clip(ori_image,t,b)

    # 除了黑色背景外的区域, 都要进行标准化
    image_nonzero = slice[np.nonzero(slice)]
    # image_nonzero = slice

    if np.std(slice) == 0 or np.std(image_nonzero) == 0:
        return slice
    else:
        normalize_image = (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
        return normalize_image
    
def closest_power_of_two(size: Union[int, Tuple[int, int]]) -> Union[int, Tuple[int, int]]:
    """
    根据输入的图片大小, 返回最接近的2的幂次大小。

    参数:
        size (int 或 tuple): 输入的图片尺寸。如果是整数，则表示宽和高相同；如果是元组，则表示宽和高分别为不同的值。

    返回:
        int 或 tuple: 最接近的2的幂次大小。

    示例:
        >>> closest_power_of_two(100)
        128
        >>> closest_power_of_two((100, 200))
        (128, 256)
    """
    def calculate_closest(n: int) -> int:
        """
        计算最接近的2的幂次
        """
        if n <= 0:
            return 0
        log2_n = math.log2(n)
        floor = math.floor(log2_n)
        ceil = math.ceil(log2_n)
        if abs(n - 2 ** floor) < abs(n - 2 ** ceil):
            return 2 ** floor
        else:
            return 2 ** ceil

    if isinstance(size, tuple):
        return tuple(calculate_closest(dim) for dim in size)
    else:
        return calculate_closest(size)
