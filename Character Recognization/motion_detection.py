import cv2
import numpy as np
import os

def plate_detect(ori_img):
   img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2GRAY)
   height = img.shape[0]
   width = img.shape[1]
   _, mask = cv2.threshold(img,thresh=200,maxval=255,type=cv2.THRESH_BINARY+cv2.THRESH_OTSU)
   img_mask = cv2.bitwise_and(img, mask)
   # yello case
   # hsv = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV)
   # cv2.imshow('hsv',hsv)
   # h, s, v1 = cv2.split(hsv)
   # lower_white = np.array([20, 100, 100], dtype=np.uint8)
   # cv2.imshow('lw',lower_white)
   # upper_white = np.array([30, 255, 255], dtype=np.uint8)
   # res_mask = cv2.inRange(hsv, lower_white, upper_white)
   # res_img = cv2.bitwise_and(v1,img, mask=res_mask)
   # cv2.imshow('res img',res_img)
   # yello case end
   #white case
   hsv = cv2.cvtColor(ori_img, cv2.COLOR_BGR2HSV)
   h, s, v1 = cv2.split(hsv)
   lower_white = np.array([0,0,160], dtype=np.uint8)
   upper_white = np.array([255,40,255], dtype=np.uint8)
   res_mask = cv2.inRange(hsv, lower_white, upper_white)
   res_img = cv2.bitwise_and(v1,img, mask=res_mask)
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
   return crop
def main():
   #window_name="Cam feed"
   #cv2.namedWindow(window_name)
   directory = '/home/jap/Desktop/'
   os.chdir(directory)
   filename = '/home/jap/Desktop/CampusSecurity/LicensePlateDetector/video12.mp4'
   cap = cv2.VideoCapture(filename)
   # cap=cv2.VideoCapture(0)


   #filename = 'F:\sample.avi'
   #codec=cv2.VideoWriter_fourcc('X','V','I','D')
   #framerate=30
   #resolution = (500,500)

 #  VideoFileOutput = cv2.VideoWriter(filename,codec,framerate,resolution)
   ret,frame1 = cap.read()
   ret,frame2 = cap.read()
   count = 1
   while ret:
      count = count+1
      ret,frame = cap.read()
      #VideoFileOutput.write(frame)

      d=cv2.absdiff(frame1,frame2)

      grey=cv2.cvtColor(d,cv2.COLOR_BGR2GRAY)

      blur =cv2.GaussianBlur(grey,(5,5),0)
      ret,th=cv2.threshold(blur,20,255,cv2.THRESH_BINARY)
      dilated=cv2.dilate(th,np.ones((3,3),np.uint8),iterations=3)
      c,h=cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
      car_detected = False
      for i,contour in enumerate(c):
         c_area = cv2.contourArea(contour)
         if c_area > 10000:
            car_detected = True
      # if not c:
      #    print(1)
      if(car_detected == True):
         number_plate = plate_detect(frame1)
         cv2.imwrite(str(count)+'.jpg',number_plate)
         cv2.drawContours(frame1,c,-1,(0,255,0),2)   
      #cv2.imshow("win1",frame2)
      cv2.imshow("inter",frame1)
      
      if cv2.waitKey(40) == 27:
         break
      frame1 = frame2
      ret,frame2= cap.read()
   cv2.destroyAllWindows()
   #VideoFileOutput.release()
   cap.release()
main()   

    
