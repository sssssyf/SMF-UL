import scipy.io as sio
import cv2
from tqdm import tqdm
import numpy as np
import time

path = 'D:/HSI_data/'
img = sio.loadmat(path + 'WHU_Hi_LongKou.mat')
img = img['WHU_Hi_LongKou']

spec = img.copy()
spec = spec / spec.max()
m, n, b = img.shape
feature_time1 = time.time()

for i in tqdm(range(3, b)):
    hsv = np.zeros((m, n, 3))
    x1 = img[:, :, i - 3:i]
    x2 = img[:, :, i - 2:i + 1]
    hsv[..., 0] = cv2.normalize(x2[..., 0], None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 1] = cv2.normalize(x2[..., 1], None, 0, 255, cv2.NORM_MINMAX)
    hsv[..., 2] = cv2.normalize(x2[..., 2], None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    cv2.imwrite('./HSImage/WHU_Hi_LongKou/' + str(i) + '.png', bgr)  # 存储为假彩色图像




