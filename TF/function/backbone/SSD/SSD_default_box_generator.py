#https://github.com/yeomko22/ssd_defaultbox_generator/blob/master/ssd_defaultbox_generator.ipynb

import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys
import random

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

def get_center(Fk):
    """
    :param Fk: feature map size
    :return: center index for feature map k
    """
    centers = []
    for i in range(Fk):
        for j in range(Fk):
            i_val = round(((i + 0.5)/Fk), 2)
            j_val = round(((j + 0.5)/Fk), 2)
            centers.append((i_val, j_val))
    return centers

Fk = 5
centers = get_center(Fk)
print(f'feature map size: {Fk}x{Fk}')
print(f'total indexes: {len(centers)}')
print(centers)

def center_crop(img, target_h, target_w):
    """
    load image, crop into 300x300 size and plot
    """
    h, w = img.shape[:2]
    mid_h, mid_w = h//2, w//2
    offset_h, offset_w = target_h//2, target_w//2
    img = img[mid_h-offset_h:mid_h+offset_h, mid_w-offset_w:mid_w+offset_w]
    return img

def show_img(img, title):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.show()
    plt.close()

img = cv2.imread('./car.jpg', cv2.IMREAD_COLOR)
img = center_crop(img, 300, 300)
show_img(img, title='origin')

def plot_centers(img, centers):
    """
    plot center points, supporting k feature map size is 5
    """
    print(img.shape)
    w, h = img.shape[:2]
    for center in centers:
        coords = (int(w * center[0]), int(h * center[1]))
        cv2.circle(img, coords, 3, (0, 0, 255), -1)

centers = get_center(5)
plot_centers(img, centers)
show_img(img, 'centers')

def plot_default_boxes(img, center, width_height):
    """
    pick one center and draw default boxes supposing k=3
    """
    cen_x, cen_y = center
    w, h = img.shape[:2]
    for w_h in width_height:
        box_w, box_h = w_h
        start = (int(w * (cen_x - (box_w))), int(h * (cen_y - box_h)))
        end = (int(w * (cen_x + (box_w))), int(h * (cen_y + box_h)))
        print(f'start: {start} end: {end}')
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img, start, end, color, 2)

center = centers[12]
width_height = width_heights[2]
plot_default_boxes(img, center, width_height)
show_img(img, 'default boxes')