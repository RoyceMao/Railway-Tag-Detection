# -*- coding: utf-8 -*-
"""
Created on 2018/11/7 10:34

@author: royce.mao

# 构造第2阶段，小图片数字的识别检测网络
"""
from keras import layers
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, BatchNormalization, Reshape, concatenate
from keras_applications.resnet50 import identity_block, conv_block
import tensorflow as tf

def stage_2_net(nb_classes, input_tensor, height=80, width=40):
    """
    自己设计的4倍下采样，简易类VGG基础网络
    :return: 
    """
    bn_axis = 3
    # net卷积,图尺寸不变
    conv1 = Convolution2D(filters=16, kernel_size=3, strides=1, padding='same')(input_tensor)
    act1 = Activation('relu')(conv1)
    bn1 = BatchNormalization(axis=bn_axis)(act1)
    # net卷积及4倍下采样
    conv2 = Convolution2D(filters=16, kernel_size=3, strides=2, padding='same')(bn1)
    act2 = Activation('relu')(conv2)
    bn2 = BatchNormalization(axis=bn_axis)(act2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)
    '''
    # 特征融合
    concat = concatenate([conv2, pool2])
    '''
    # 接上cls输出层
    classification = Convolution2D(filters=height * width * 9 * nb_classes // 16, kernel_size=3, padding='same')(pool2)
    classification = Reshape(target_shape=(-1, nb_classes))(classification)
    classification = Activation(activation='softmax', name='classification')(classification)
    # 接上regr输出层
    bboxes_regression = Convolution2D(filters=height * width * 9 * 4 // 16, kernel_size=3, padding='same')(pool2)
    bboxes_regression = Reshape(target_shape=(-1, nb_classes-1), name='regression')(bboxes_regression)
    '''
    # 接上cls输出层
    classification = Dense(nb_classes, activation='softmax', kernel_initializer='zero')(cat)
    # 接上regr输出层
    bboxes_regression = Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero')(cat)
    '''
    # 最终的model构建
    # detect_model = Model(inputs=input_tensor, outputs=[classification, bboxes_regression])
    # detect_model.summary()

    return [classification, bboxes_regression]

if __name__ == "__main__":

    stage_2_net(40,40,11)
