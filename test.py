'''
Description: 
Author: notplus
Date: 2021-11-18 10:28:22
LastEditors: notplus
LastEditTime: 2021-11-19 09:30:43
FilePath: /test.py

Copyright (c) 2021 notplus
'''

import tensorflow as tf
import os
import numpy as np
import cv2
from pfld import PFLD

# Load checkpoint
# model = PFLD()
# model(tf.random.normal((1, 112, 112, 3)))
# model.load_weights('./weight/wing_loss/pfld_wing_loss')

# Load saved model
model = tf.keras.models.load_model('./weight/pfld_wing_loss')

for root, dirs, files in os.walk('./test'):
    for f in files:
        image_path = os.path.join(root, f)
        print(image_path)

        img_origin = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img_origin, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        img = img_rgb.astype(np.float32)
        img -= 128.0
        img /= 128.0
        src = cv2.resize(img, (112, 112)).reshape(-1, 112, 112, 3)

        land_marks = model(src).numpy().reshape(-1) * h

        # img = cv2.resize(img_origin, (112, 112))
        img = img_origin
        for i in range(98):
            img = cv2.circle(img, (land_marks[i*2], land_marks[i*2+1]), 1, color=(0, 0, 255),thickness=-1)
        # img = cv2.resize(img, (512, 512))
        cv2.imshow("tmp", img)
        cv2.waitKey(0)

        
        