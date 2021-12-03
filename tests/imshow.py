import cv2
import matplotlib.pyplot as plt
import numpy as np


img1 = cv2.imread('/home/hades/Desktop/Tool/data/05_ggg05_s02_010_r2S_T_A_CHS_NON/annotations/sketch_0001.png')
img2 = cv2.imread('/home/hades/Desktop/Tool/data/05_ggg05_s02_010_r2S_T_A_CHS_NON/annotations/sketch_0002.png')
print(np.unique(img1))
img = np.hstack((img1, img2))
plt.imshow(img)
plt.show()