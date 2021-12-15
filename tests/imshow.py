import cv2
import matplotlib.pyplot as plt
import numpy as np


img1 = cv2.imread('/home/hades/Desktop/Tool/data/41_ggg41_s17_002_RS_C_CHS_NON/annotations/sketch_0001.png')
img2 = cv2.imread('/home/hades/Desktop/Tool/data/41_ggg41_s17_002_RS_C_CHS_NON/annotations/sketch_0003.png')
print(np.unique(img1))
img = np.hstack((img1, img2))
plt.imshow(img)
plt.show()