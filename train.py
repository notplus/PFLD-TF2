'''
Description: 
Author: notplus
Date: 2021-11-18 10:28:12
LastEditors: notplus
LastEditTime: 2021-11-18 14:51:06
FilePath: /train.py

Copyright (c) 2021 notplus
'''


import tensorflow as tf
import matplotlib.pyplot as plt

from pfld import PFLD
import tfrecord
import config as cfg

train_dataset = tfrecord.get_dataset(cfg.TRAIN_TFREC, cfg.BATCH_SIZE)
val_dataset = tfrecord.get_dataset(cfg.VAL_TFREC, cfg.BATCH_SIZE)

name = "pfld"

model = PFLD(summary=True)

# Callbacks
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    "./weight/" + name, save_best_only=True
)
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=20, restore_best_weights=True
)
csv_logger_cb = tf.keras.callbacks.CSVLogger("./log/" + name + ".csv")

## Complile model
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, decay_steps=20000, alpha=0.0
)
# opt = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9, decay=1e-4)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)#, amsgrad=True)
model.compile(optimizer=opt)

hist = model.fit(
    train_dataset,
    epochs=250,
    validation_data=val_dataset,
    callbacks=[checkpoint_cb, early_stopping_cb, csv_logger_cb]
)

print('Trainig finished.')

# Loss History
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('rate')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig('./log/loss.jpg')