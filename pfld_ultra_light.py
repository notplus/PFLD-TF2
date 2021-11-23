'''
Description: 
Author: notplus
Date: 2021-11-22 10:30:00
LastEditors: notplus
LastEditTime: 2021-11-23 16:14:12
FilePath: /pfld_ultra_light.py

Copyright (c) 2021 notplus
'''

import math

from tensorflow.python.ops.gen_math_ops import mul
import losses
import tensorflow as tf
import tensorflow.keras.layers as layers

def _conv_block(filters, kernel_size, strides, padding, group=1, has_bn=True, is_linear=False):
    return tf.keras.Sequential([
        layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                      groups=group, use_bias=False),
        layers.BatchNormalization() if has_bn else tf.keras.Sequential(),
        layers.ReLU() if not is_linear else tf.keras.Sequential()
    ])

class GhostModule(layers.Layer):
    def __init__(self, filters, is_linear=False):
        super().__init__()

        self.filters = filters
        init_channel = math.ceil(filters / 2)
        new_channel = init_channel

        self.primary_conv = _conv_block(init_channel, 1, 1, padding='valid', is_linear=is_linear)
        self.cheap_operation = _conv_block(new_channel, 3, 1, padding='same', group=init_channel, is_linear=is_linear)
        
    def call(self, inputs):
        x1 = self.primary_conv(inputs)
        x2 = self.cheap_operation(x1)
        out = layers.concatenate([x1, x2], axis=-1)
        return out[:, :, :, :self.filters]


class GhostBottleneck(layers.Layer):
    def __init__(self, in_channel, hidden_channel, out_channel, stride):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.ghost_conv = tf.keras.Sequential([
            # GhostModule
            GhostModule(hidden_channel, is_linear=False),
            # DepthwiseConv-linear
            _conv_block(hidden_channel, 3, stride, padding='same', group=hidden_channel, is_linear=True) if stride == 2 else tf.keras.Sequential(),
            # layers.DepthwiseConv2D(hidden_channel, stride, padding='same') if stride == 2 else tf.keras.Sequential(),
            # GhostModule-linear
            GhostModule(out_channel, is_linear=True)
        ])

        if stride == 1 and in_channel == out_channel:
            self.shortcut = tf.keras.Sequential()
        else:
            self.shortcut = tf.keras.Sequential([
                _conv_block(in_channel, 3, stride, padding='same', group=in_channel, is_linear=True),
                _conv_block(out_channel, 1, 1, padding='valid', is_linear=True)
            ])

    def call(self, x):
        return self.ghost_conv(x) + self.shortcut(x)


class PFLD_Ultralight(tf.keras.Model):
    def __init__(self, width_factor=1, input_size=112, landmark_number=98):
        super(PFLD_Ultralight, self).__init__()

        self.conv1 = _conv_block(int(64 * width_factor), 3, 2, padding='same')
        self.conv2 = _conv_block(int(64 * width_factor), 3, 1, padding='same', group=int(64 * width_factor))

        self.conv3_1 = GhostBottleneck(int(64 * width_factor), int(128 * width_factor), int(80 * width_factor), stride=2)
        self.conv3_2 = GhostBottleneck(int(80 * width_factor), int(160 * width_factor), int(80 * width_factor), stride=1)
        self.conv3_3 = GhostBottleneck(int(80 * width_factor), int(160 * width_factor), int(80 * width_factor), stride=1)

        self.conv4_1 = GhostBottleneck(int(80 * width_factor), int(240 * width_factor), int(96 * width_factor), stride=2)
        self.conv4_2 = GhostBottleneck(int(96 * width_factor), int(288 * width_factor), int(96 * width_factor), stride=1)
        self.conv4_3 = GhostBottleneck(int(96 * width_factor), int(288 * width_factor), int(96 * width_factor), stride=1)

        self.conv5_1 = GhostBottleneck(int(96 * width_factor), int(384 * width_factor), int(144 * width_factor), stride=2)
        self.conv5_2 = GhostBottleneck(int(144 * width_factor), int(576 * width_factor), int(144 * width_factor), stride=1)
        self.conv5_3 = GhostBottleneck(int(144 * width_factor), int(576 * width_factor), int(144 * width_factor), stride=1)
        self.conv5_4 = GhostBottleneck(int(144 * width_factor), int(576 * width_factor), int(144 * width_factor), stride=1)

        self.conv6 = GhostBottleneck(int(144 * width_factor), int(288 * width_factor), int(16 * width_factor), stride=1)
        self.conv7 = _conv_block(int(32 * width_factor), 3, 1, padding='same')
        self.conv8 = _conv_block(int(128 * width_factor), input_size // 16, 1, padding='valid', has_bn=False)

        self.avg_pool1 = layers.AvgPool2D(input_size // 2)
        self.avg_pool2 = layers.AvgPool2D(input_size // 4)
        self.avg_pool3 = layers.AvgPool2D(input_size // 8)
        self.avg_pool4 = layers.AvgPool2D(input_size // 16)

        self.fc = layers.Dense(landmark_number * 2)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.avg_pool1(x)
        x1 = layers.Flatten()(x1)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x2 = self.avg_pool2(x)
        x2 = layers.Flatten()(x2)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x3 = self.avg_pool3(x)
        x3 = layers.Flatten()(x3)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x4 = self.avg_pool4(x)
        x4 = layers.Flatten()(x4)

        x = self.conv6(x)
        x = self.conv7(x)
        x5 = self.conv8(x)
        x5 = layers.Flatten()(x5)

        multi_scale = layers.concatenate([x1, x2, x3, x4, x5], axis=-1)
        landmarks = self.fc(multi_scale)

        return landmarks

    def train_step(self, data):
        img_tensor, _, landmark_gt, _ = data
        
        with tf.GradientTape() as tape:
            landmarks = self(img_tensor, training=True)  # Forward pass

            loss = losses.wing_loss(landmark_gt, landmarks)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.loss_tracker.update_state(loss)
        # Return a dict mapping metric names to current value
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        img_tensor, _, landmark_gt, _ = data
        
        landmarks = self(img_tensor, training=False)  # Forward pass
        
        loss = tf.reduce_mean(tf.reduce_sum((landmark_gt - landmarks) * (landmark_gt - landmarks)))

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.loss_tracker]

if __name__ == '__main__':
    model = PFLD_Ultralight()

    model(tf.random.normal((1, 112, 112, 3)))
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
