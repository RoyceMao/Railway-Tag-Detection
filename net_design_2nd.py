# -*- coding: utf-8 -*-
"""
Created on 2018/11/7 10:34

@author: royce.mao

# 构造第2阶段，小图片数字的识别检测网络
"""
from keras import layers
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, BatchNormalization, Reshape, concatenate, Dense
from keras_applications.resnet50 import identity_block, conv_block
import tensorflow as tf

def stage_2_net(nb_classes, input_tensor, height=80, width=40):
    """
    自己设计的8倍下采样，简易类VGG基础网络
    :return: 
    """
    bn_axis = 3
    # 只有net卷积的8倍下采样
    conv1_1 = Convolution2D(filters=16, kernel_size=3, strides=2, padding='same')(input_tensor)
    bn1_1 = BatchNormalization(axis=bn_axis)(conv1_1)
    act1_1 = Activation('relu')(bn1_1)
    conv1_2 = Convolution2D(filters=16, kernel_size=3, strides=2, padding='same')(act1_1)
    bn1_2 = BatchNormalization(axis=bn_axis)(conv1_2)
    act1_2 = Activation('relu')(bn1_2)
    conv1_3 = Convolution2D(filters=16, kernel_size=3, strides=2, padding='same')(act1_2)
    # 2倍net卷积加4倍pooling的总共8倍下采样
    conv2_1 = Convolution2D(filters=16, kernel_size=3, strides=2, padding='same')(input_tensor)
    bn2_1 = BatchNormalization(axis=bn_axis)(conv2_1)
    act2_1 = Activation('relu')(bn2_1)
    pool2_1 = MaxPooling2D(pool_size=(4, 4))(act2_1)
    # 4倍net卷积加2倍pooling的总共8倍下采样
    conv3_1 = Convolution2D(filters=16, kernel_size=3, strides=2, padding='same')(input_tensor)
    bn3_1 = BatchNormalization(axis=bn_axis)(conv3_1)
    act3_1 = Activation('relu')(bn3_1)
    conv3_2 = Convolution2D(filters=16, kernel_size=3, strides=2, padding='same')(act3_1)
    bn3_2 = BatchNormalization(axis=bn_axis)(conv3_2)
    act3_2 = Activation('relu')(bn3_2)
    pool3_1 = MaxPooling2D(pool_size=(2, 2))(act3_2)
    # 特征融合
    concat = concatenate([conv1_3, pool2_1, pool3_1])

    # 接上cls输出层
    classification = Convolution2D(filters=height * width * 15 * nb_classes // 3200, kernel_size=3, padding='same')(concat)
    classification = Reshape(target_shape=(-1, nb_classes))(classification)
    classification = Activation(activation='softmax', name='classification')(classification)
    # 接上regr输出层
    bboxes_regression = Convolution2D(filters=height * width * 15 * 4 // 320, kernel_size=3, padding='same')(concat)
    bboxes_regression = Reshape(target_shape=(-1, 4*(nb_classes-1)), name='regression')(bboxes_regression)
    '''
    # 接上cls输出层
    classification = Dense(nb_classes, activation='softmax', kernel_initializer='zero')(concat)
    # 接上regr输出层
    bboxes_regression = Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero')(concat)
    '''
    # 最终的model构建
    # detect_model = Model(inputs=input_tensor, outputs=[classification, bboxes_regression])
    # detect_model.summary()

    return [classification, bboxes_regression]

if __name__ == "__main__":
    stage_2_net(11, Input(shape=(40, 20, 3)), height=40, width=20)
