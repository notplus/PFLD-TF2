'''
Description: 
Author: notplus
Date: 2021-11-22 18:17:25
LastEditors: notplus
LastEditTime: 2021-11-23 16:12:56
FilePath: /camera.py

'''

import cv2
import numpy as np
import tensorflow as tf

from pfld_ultra_light import PFLD_Ultralight
from mtcnn.detector import detect_faces


model_path = './weight/pfld_ultra/'

model = PFLD_Ultralight()
model(tf.random.normal((1, 112, 112, 3)))
model.load_weights(model_path)

# model = tf.keras.models.load_model('./weight/pfld_wing_loss')

cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    if not ret: break
    height, width = img.shape[:2]
    bounding_boxes, landmarks = detect_faces(img)
    for box in bounding_boxes:
        x1, y1, x2, y2 = (box[:4] + 0.5).astype(np.int32)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        w = x2 - x1 + 1
        h = y2 - y1 + 1
        cx = x1 + w // 2
        cy = y1 + h // 2

        size = int(max([w, h]) * 1.1)
        x1 = cx - size // 2
        x2 = x1 + size
        y1 = cy - size // 2
        y2 = y1 + size

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)

        edx1 = max(0, -x1)
        edy1 = max(0, -y1)
        edx2 = max(0, x2 - width)
        edy2 = max(0, y2 - height)

        cropped = img[y1:y2, x1:x2]
        if (edx1 > 0 or edy1 > 0 or edx2 > 0 or edy2 > 0):
            cropped = cv2.copyMakeBorder(cropped, edy1, edy2, edx1, edx2,
                                            cv2.BORDER_CONSTANT, 0)
        input_img = cv2.resize(cropped, (112, 112))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.float32)
        input_img -= 128.
        input_img /= 128.

        landmarks = model(input_img[None, ...])
        pre_landmark = landmarks[0]
        pre_landmark = pre_landmark.numpy().reshape(
            -1, 2) * [size, size] - [edx1, edy1]

        for (x, y) in pre_landmark.astype(np.int32):
            cv2.circle(img, (x1 + x, y1 + y), 2, (0, 0, 255))

    cv2.imshow('face_landmark_98', img)
    if cv2.waitKey(10) == 27:
        break
