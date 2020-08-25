import cv2
import numpy as np
from tkinter.filedialog import askopenfilename
filename = askopenfilename()    
ori_img = cv2.imread(filename,cv2.IMREAD_COLOR)
img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
height = img.shape[0]
width = img.shape[1]
# kernel = np.ones((2, 2),np.uint8)
_, mask = cv2.threshold(img,thresh=200,maxval=255,type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('mask',mask)
img_mask = cv2.bitwise_and(img, mask)
cv2.imshow('img masked',img_mask)
# yello case
hsv = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV)
cv2.imshow('hsv',hsv)
h, s, v1 = cv2.split(hsv)
lower_white = np.array([20, 100, 100], dtype=np.uint8)
cv2.imshow('lw',lower_white)
upper_white = np.array([30, 255, 255], dtype=np.uint8)
res_mask = cv2.inRange(hsv, lower_white, upper_white)
res_img = cv2.bitwise_and(v1,img, mask=res_mask)
cv2.imshow('res img',res_img)
# yello case end
#white case
# hsv = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV)
# cv2.imshow('hsv',hsv)
# h, s, v1 = cv2.split(hsv)
# lower_white = np.array([0,0,160], dtype=np.uint8)
# cv2.imshow('lw',lower_white)
# upper_white = np.array([255,40,255], dtype=np.uint8)
# res_mask = cv2.inRange(hsv, lower_white, upper_white)
# res_img = cv2.bitwise_and(v1,img, mask=res_mask)
# cv2.imshow('res img',res_img)
#white case end
edges = cv2.Canny(res_img, height, width)
contours, _ = cv2.findContours(res_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]
NumberPlateCnt = None
found = False
lt, rb = [10000, 10000], [0, 0]
if len(contours) > 0:
	hull = cv2.convexHull(contours[0])
	approx2 = cv2.approxPolyDP(hull,0.01*cv2.arcLength(hull,True),True)
	cv2.drawContours(ori_img, [approx2], -1, (255, 0, 255), 2, lineType=8)
	for point in approx2:
		cur_cx, cur_cy = point[0][0], point[0][1]
		if cur_cx < lt[0]: lt[0] = cur_cx
		if cur_cx > rb[0]: rb[0] = cur_cx
		if cur_cy < lt[1]: lt[1] = cur_cy
		if cur_cy > rb[1]: rb[1] = cur_cy
	cv2.circle(ori_img, (lt[0], lt[1]), 2, (150, 200, 255), 2)
	cv2.circle(ori_img, (rb[0], rb[1]), 2, (150, 200, 255), 2)
	crop = res_img[lt[1]:rb[1], lt[0]:rb[0]]
	crop_res = ori_img[lt[1]:rb[1], lt[0]:rb[0]]
else:
	crop = res_img.copy()
	crop_res = ori_img.copy()
cv2.imshow('crop',crop)
cv2.imshow('ori_img',ori_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
	
    
