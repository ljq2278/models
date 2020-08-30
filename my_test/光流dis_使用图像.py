import cv2 as cv
import numpy as np


def show_flow_hsv(flow, show_style=1):
   mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])  # 将直角坐标系光流场转成极坐标系

   hsv = np.zeros((flow.shape[0], flow.shape[1], 3), np.uint8)

   # 光流可视化的颜色模式
   if show_style == 1:
      hsv[..., 0] = ang * 180 / np.pi / 2  # angle弧度转角度
      hsv[..., 1] = 255
      hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)  # magnitude归到0～255之间
   elif show_style == 2:
      hsv[..., 0] = ang * 180 / np.pi / 2
      hsv[..., 1] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
      hsv[..., 2] = 255

   # hsv转bgr
   bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
   return bgr


img1 = cv.imread(r"D:\dataset\traffic_state_predict\amap_traffic_train_12cate\000030_1_2019-04-02-08-38-44.jpg")#导入灰度图像
img2 = cv.imread(r"D:\dataset\traffic_state_predict\amap_traffic_train_12cate\000030_2_2019-04-02-08-38-49.jpg")

prvs = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
hsv = np.zeros_like(img1)
hsv[...,1] = 255
dis = cv.DISOpticalFlow_create(1)

next = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
flow = dis.calc(prvs,next, None,)
#########################################################################
# mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
# hsv[...,0] = ang*180/np.pi/2
# hsv[...,2] = cv.normalize(mag,None,0,255,cv.NORM_MINMAX)
# bgr = cv.cvtColor(hsv,cv.COLOR_HSV2BGR)
#########################################################################
bgr = show_flow_hsv(flow)
cv.imshow('result',bgr)
cv.imshow('input', img2)
k = cv.waitKey(30) & 0xff

cv.destroyAllWindows()