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

def stage_2_net(nb_classes, input_tensor, height=40, width=20):
    """
    自己设计的4倍下采样，简易类VGG基础网络
    :return: 
    """
    bn_axis = 3
    # 只有net卷积的4倍下采样
    conv1 = Convolution2D(filters=16, kernel_size=3, strides=4, padding='same')(input_tensor)
    act1 = Activation('relu')(conv1)
    bn1 = BatchNormalization(axis=bn_axis)(act1)
    # net卷积加pooling的4倍下采样
    conv2 = Convolution2D(filters=16, kernel_size=3, strides=2, padding='same')(input_tensor)
    act2 = Activation('relu')(conv2)
    bn2 = BatchNormalization(axis=bn_axis)(act2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(bn2)
    # 只有pooling的4倍下采样
    pool3 = MaxPooling2D(pool_size=(4, 4))(input_tensor)
    # 特征融合
    concat = concatenate([conv1, pool2, pool3])

    # 接上cls输出层
    classification = Convolution2D(filters=height * width * 3 * nb_classes // 16, kernel_size=3, padding='same')(concat)
    classification = Reshape(target_shape=(-1, nb_classes))(classification)
    classification = Activation(activation='softmax', name='classification')(classification)
    # 接上regr输出层
    bboxes_regression = Convolution2D(filters=height * width * 3 * 4 // 16, kernel_size=3, padding='same')(concat)
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
