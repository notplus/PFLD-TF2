'''
Description: 
Author: notplus
Date: 2021-11-18 10:28:55
LastEditors: notplus
LastEditTime: 2021-11-18 15:57:19
FilePath: /tfrecord.py

Copyright (c) 2021 notplus
'''

import tensorflow as tf
import config as cfg

def parse_tfrecord_fn(serial_exmp):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "attribute": tf.io.FixedLenFeature([], tf.string),
        "landmark": tf.io.FixedLenFeature([], tf.string),
        "euler_angle": tf.io.FixedLenFeature([], tf.string),
    }

    feats = tf.io.parse_single_example(serial_exmp, feature_description)
    feats["image"] = tf.io.decode_jpeg(feats["image"], channels=3)
    feats["attribute"] = tf.io.decode_raw(feats["attribute"], tf.float32)
    feats["landmark"] = tf.io.decode_raw(feats["landmark"], tf.float32)
    feats["euler_angle"] = tf.io.decode_raw(feats["euler_angle"], tf.float32)

    feats["attribute"] = tf.reshape(feats["attribute"], [6])
    feats["landmark"] = tf.reshape(feats["landmark"], [196])
    feats["euler_angle"] = tf.reshape(feats["euler_angle"], [3])

    return feats


def prepare_sample(features):
    image = tf.cast(features["image"], tf.float32)
    image -= 128.0  # mean
    image /= 128.0  # std

    return image, features["attribute"], features["landmark"], features["euler_angle"]


def get_dataset(filename, batch_size):
    dataset = (
        tf.data.TFRecordDataset(filename)
        .map(parse_tfrecord_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .map(prepare_sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .shuffle(batch_size * 32)
        .batch(batch_size)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )
    return dataset



if __name__ == "__main__":
    print(cfg.TRAIN_TFREC)
    dataset = get_dataset(cfg.TRAIN_TFREC, 1)
    
    for features in dataset.take(75000):
        img_tensor, attribute_gt, landmark_gt, euler_angle_gt = features
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(img_tensor.numpy().reshape((112, 112, 3)))
        # print(attribute_gt.shape)

        # from losses import loss_fn
        # angle = tf.random.normal((1, 3))
        # landmarks = tf.random.normal((1, 196))
        # weighted_loss, loss = loss_fn(attribute_gt, landmark_gt, euler_angle_gt, angle, landmarks)
        # print(weighted_loss)
        # print(loss)
