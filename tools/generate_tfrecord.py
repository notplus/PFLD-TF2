'''
Description: 
Author: notplus
Date: 2021-11-18 10:28:55
LastEditors: notplus
LastEditTime: 2021-11-18 15:59:42
FilePath: /tools/generate_tfrecord.py

Copyright (c) 2021 notplus
'''

import tensorflow as tf
import config as cfg
import numpy as np

def image_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(value).numpy()]))


def float_feature_list(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def float_feature_np(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.tobytes()]))

def create_example(image, attribute, landmark, euler_angle):
    feature = {
        "image": image_feature(image),
        "attribute": float_feature_np(attribute),
        "landmark": float_feature_np(landmark),
        "euler_angle": float_feature_np(euler_angle),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))

def write_tf_record(writer, file_list):
    with open(file_list, 'r') as f:
        for line in f.readlines():
            line_split = line.strip().split()
            
            img_tensor = tf.io.decode_jpeg(tf.io.read_file(line_split[0]))
            landmark = np.asarray(line_split[1:197], dtype=np.float32)
            attribute = np.asarray(line_split[197:203], dtype=np.float32)
            euler_angle = np.asarray(line_split[203:206], dtype=np.float32)
            
            example = create_example(img_tensor, attribute, landmark, euler_angle)
            writer.write(example.SerializeToString())

if __name__ == '__main__':
    train_file_list = './data/train_data/list.txt'
    test_file_list = './data/test_data/list.txt'
    
    with tf.io.TFRecordWriter(cfg.TRAIN_TFREC) as writer:
        write_tf_record(writer, train_file_list)

    with tf.io.TFRecordWriter(cfg.VAL_TFREC) as writer:
        write_tf_record(writer, test_file_list)