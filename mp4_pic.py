#!/usr/bin/env python
# -*- coding: utf-8 -*-


import cv2


file_path=r'\\192.168.25.73\Team-CV\20190701-Mousehole-maodeng\2019_07_01_16_33_38.mp4'
vc = cv2.VideoCapture(file_path)  # 读入视频文件
d = 0
exit_flag = False
c = 0
images = []
images_origin = []

savepath=file_path[:-4]

index=0
while True:  # 循环读取视频帧
    rval, image = vc.read()
    index+=1
    print(index)
    cv2.imwrite('shudong/%05d' % index+".jpg",image)