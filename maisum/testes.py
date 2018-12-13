# -*- coding: utf-8 -*-
import cv2
import numpy as np


print (cv2.__version__)
img = cv2.imread('datasets/figura.jpg',0)

#while (1):
bbox = cv2.selectROI(img, False)

p1 = (int(bbox[0]), int(bbox[1]))
p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
cv2.rectangle(img, p1, p2, (0,0,255))
x1 = int(bbox[0])
y1 = int(bbox[1])
x2 = int(bbox[0]) + int(bbox[2])
y2 = int(bbox[1]) + int(bbox[3])
print (img.shape)
print (bbox)
img = img[y1:y2,x1:x2]
cv2.imshow('image',img)
cv2.waitKey(0)

cv2.destroyAllWindows()