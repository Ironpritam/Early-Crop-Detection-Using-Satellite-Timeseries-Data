import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight


class Inception(tf.keras.Model):
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        self.b1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu', name='b1_conv1x1')

        self.b2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu', name='b2_conv1x1')
        self.b2_2 = tf.keras.layers.Conv2D(c2[1], 3, padding='same', activation='relu', name='b2_conv3x3')

        self.b3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu', name='b3_conv1x1')
        self.b3_2 = tf.keras.layers.Conv2D(c3[1], 5, padding='same', activation='relu', name='b3_conv5x5')

        self.b4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same', name='b4_maxpool')
        self.b4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu', name='b4_conv1x1')

    def call(self, x):
        b1 = self.b1_1(x)
        b2 = self.b2_2(self.b2_1(x))
        b3 = self.b3_2(self.b3_1(x))
        b4 = self.b4_2(self.b4_1(x))
        return tf.keras.layers.Concatenate()([b1, b2, b3, b4])

# Funtion to get grad-cam filter to apply on original image to highlight features.


# Building classifier model using convulution layers.
class Crop_BB():
    def b1(self):
        return tf.keras.Sequential([
              tf.keras.layers.Conv2D(16, 1, activation='relu', name='b1_conv1x1'),
              tf.keras.layers.Conv2D(32 3, padding='same', activation='relu', name='b1_conv3x3'),
              tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same', name='b1_maxpool')])
    def b2(self):
        return tf.keras.Sequential([
            Inception(32, (17, 32), (8, 8), 8),
            tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same', name='b2_maxpool')])
    def b3(self):
        return tf.keras.Sequential([
            Inception(32, (20, 40), (8, 16), 16),
            Inception(48, (48, 96), (12, 32), 32),
            tf.keras.layers.Conv2D(32, 1, activation='relu', name='b3_conv1x1'),
            tf.keras.layers.GlobalAveragePooling2D(name='global_avg_pool'),
            tf.keras.layers.Dense(9, activation='relu', name='dense_output'),
            tf.keras.layers.Dense(3, activation='softmax', name='dense_output')])


crops_bb = Crop_BB()
model = tf.keras.Sequential([crops_bb.b1(), crops_bb.b2(), crops_bb.b3()])