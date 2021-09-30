#https://github.com/yeomko22/ssd_defaultbox_generator/blob/master/ssd_defaultbox_generator.ipynb

import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

S_MIN = 0.2
S_MAX = 0.9

def get_scales(m):
    """
    get scale levels for each feature map k
    all values are relative ratio to input image width, height
    :param m: number of feature maps to perform object detection
    :return: scales level
    """
    scales = []
    for k in range(1, m+1):
        scales.append(round((S_MIN + (S_MAX-S_MIN) / (m-1)*(k-1)), 2))
    return scales

# print(get_scales(6))            # [0.2, 0.34, 0.48, 0.62, 0.76, 0.9]

ratios = [1, 2, 3, 0.5, 0.33]
def get_width_height(scales):
    """
    get default box width, height for feature map k
    all values are relative ration to input image width, height
    """
    width_heights = []
    for k, scale in enumerate(scales):
        print(f'k: {k+1} scale: {scale}')
        width_height_per_scale = []
        for ratio in ratios:
            width = min(round((scale * math.sqrt(ratio)), 2), 1)
            height = min(round((scale / math.sqrt(ratio)), 2), 1)
            width_height_per_scale.append((width, height))
            print(f'width: {width} height: {height}')
            if k < len(scales) -1:
                extra_scale = round(math.sqrt(scale * scales[k+1]), 2)
                width_height_per_scale.append((extra_scale, extra_scale))
            width_heights.append(width_height_per_scale)
        print(f'width: {extra_scale} height: {extra_scale}')
        print('')
    return width_heights

scales = get_scales(6)
width_heights = get_width_height(scales)
print(width_heights)
