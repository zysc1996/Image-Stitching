import numpy as np
import cv2
from matplotlib import pyplot as plt


img1 = cv2.imread('C:/Users/67444/code/image1.jpg',1)          
img2 = cv2.imread('C:/Users/67444/code/image2.jpg',1) 

#运用SIFT算法提取特征
#kp1、kp2保存了特征点的位置和大小信息，des1、des2保存了形状为128维向量的描述子
sift = cv2.xfeatures2d.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

#使用暴力匹配算法对两张图的特征点进行匹配，
bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.match(des1,des2)
 
#对错误匹配进行剔除，此处选用RANSAC算法，默认迭代次数2000次，阈值取4
src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])
homo_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3)

h1, w1 = img1.shape[0], img1.shape[1]
h2, w2 = img2.shape[0], img2.shape[1]
rect1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape((4, 1, 2))
rect2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape((4, 1, 2))
trans_rect1 = cv2.perspectiveTransform(rect1,homo_matrix)
total_rect = np.concatenate((rect2, trans_rect1), axis=0)
min_x, min_y = np.int32(total_rect.min(axis=0).ravel())
max_x, max_y = np.int32(total_rect.max(axis=0).ravel())
shift_to_zero_matrix = np.array([[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]])
trans_img1 = cv2.warpPerspective(img1, shift_to_zero_matrix.dot(homo_matrix),(max_x - min_x, max_y - min_y))


trans_img1[-min_y:h2 - min_y, -min_x:w2 - min_x] = img2



# trans_img1 = cv2.resize(trans_img1,(int(trans_img1.shape[1]/2),int(trans_img1.shape[0]/2))) 
cv2.imshow('img3',trans_img1)
key = cv2.waitKey()
if key == 27:
	cv2.destroyAllWindows()