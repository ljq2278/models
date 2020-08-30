import cv2
import matplotlib.pyplot as plt
import numpy as np


def detect(image):
    # 转化为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 创建SIFT生成器
    # descriptor是一个对象，这里使用的是SIFT算法
    descriptor = cv2.xfeatures2d.SIFT_create()
    # 检测特征点及其描述子（128维向量）
    kps, features = descriptor.detectAndCompute(image, None)
    return (kps,features)

def match_keypoints(kps_left, kps_right, features_left, features_right, ratio, threshold):
    """
    kpsA,kpsB,featureA,featureB: 两张图的特征点坐标及特征向量
    threshold: 阀值

    """
    # 建立暴力匹配器
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    # 使用knn检测，匹配left,right图的特征点
    raw_matches = matcher.knnMatch(features_left, features_right, 2)
    print(len(raw_matches))
    matches = []  # 存坐标，为了后面
    good = []  # 存对象，为了后面的演示
    # 筛选匹配点
    for m in raw_matches:
        # 筛选条件
        #         print(m[0].distance,m[1].distance)
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            good.append([m[0]])
            matches.append((m[0].queryIdx, m[0].trainIdx))
            """
            queryIdx：测试图像的特征点描述符的下标==>img_keft
            trainIdx：样本图像的特征点描述符下标==>img_right
            distance：代表这怡翠匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
            """
    # 特征点对数大于4就够用来构建变换矩阵了
    kps_left = np.float32([kp.pt for kp in kps_left])
    kps_right = np.float32([kp.pt for kp in kps_right])
    print(len(matches))
    if len(matches) > 4:
        # 获取匹配点坐标
        pts_left = np.float32([kps_left[i] for (i, _) in matches])
        pts_right = np.float32([kps_right[i] for (_, i) in matches])
        # 计算变换矩阵(采用ransac算法从pts中选择一部分点)
        H, status = cv2.findHomography(pts_right, pts_left, cv2.RANSAC, threshold)
        return (matches, H, good)
    return None

img_left = cv2.imread(r"D:\projects\tensorflowModelGarden\my_test\image_up\000008_2_2019-05-21-08-05-52_2.jpg", 1)
img_left = cv2.resize(img_left, dsize=(450, 300))
img_right = cv2.imread(r"D:\projects\tensorflowModelGarden\my_test\image_up\000008_3_2019-05-21-08-05-57.jpg", 1)
img_right = cv2.resize(img_right, dsize=(450, 300))

kps_left, features_left = detect(img_left)
kps_right, features_right = detect(img_right)
matches, H, good = match_keypoints(kps_left,kps_right,features_left,features_right,0.5,0.99)
img = cv2.drawMatchesKnn(img_left,kps_left,img_right,kps_right,good[:30],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imshow('img', img)
cv2.waitKey()

# https://blog.csdn.net/qq_44019424/article/details/106010362