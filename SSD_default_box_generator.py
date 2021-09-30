#https://github.com/yeomko22/ssd_defaultbox_generator/blob/master/ssd_defaultbox_generator.ipynb

S_MIN = 0.2
S_MAX = 0.9

def get_scales(m):
    """
    get scale levels for each feature map k
    all values are relative ratio to input image width, height
    :param m: number of feature map
    :return: scales
    """
    scales = []
    for k in range(1, m+1):
        scales.append(round((S_MIN + (S_MAX-S_MIN) / (m-1)*(k-1)), 2))
    return scales

# print(get_scales(6))            # [0.2, 0.34, 0.48, 0.62, 0.76, 0.9]

