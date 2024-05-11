# -*- coding: utf-8 -*-
"""
@Time: 2024/5/6 16:25
@Author: mengqingyao
@File: crop_face.py
"""

import dlib
import cv2
import os

# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()


def crop_face(input_folder_path, output_folder_path):
    images = os.listdir(input_folder_path)
    for image in images:
        image_path = os.path.join(input_folder_path, image)
        img = cv2.imread(image_path)

        if img is None:
            continue

        # 获取图片原始宽高
        height, width, _ = img.shape

        # 使用 Dlib 进行人脸检测
        faces = detector(img, 1)

        if len(faces) == 0:
            continue

        # 获取第一个检测到的人脸
        face = faces[0]

        # 计算面部坐标
        y1, y2, x1, x2 = face.top(), face.bottom(), face.left(), face.right()

        # 计算面部宽高
        h, w = y2 - y1, x2 - x1

        # 根据面部区域和图片边缘的距离，调整面部坐标，使面部居中并在面部周围保留不超过半张脸的最大距离，确保面部完整
        y1 -= min(int(h/2), y1)
        y2 += min(int(h/2), height-y2)
        x1 -= min(int(w/2), x1)
        x2 += min(int(w/2), width-x2)

        # 根据计算出的坐标裁剪图片
        cropped_img = img[y1:y2, x1:x2]

        # 计算最终的图片宽高比，控制宽度不超过200，高度等比例缩放
        scale = (y2 - y1) / (x2 - x1)
        wid = min(int(x2 - x1), 200)
        hei = int(wid * scale)
        resized = cv2.resize(cropped_img, (wid, hei), interpolation=cv2.INTER_AREA)

        # 输出最终图片
        output_path = os.path.join(output_folder_path, image)
        cv2.imwrite(output_path, resized)


