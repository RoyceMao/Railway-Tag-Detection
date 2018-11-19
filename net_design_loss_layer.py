# -*- coding: utf-8 -*-
"""
Created on 2018/11/7 10:34

@author: royce.mao

# 把loss写进net_design网络的layer，原第2阶段
"""
from keras import layers
from keras.models import Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, BatchNormalization, Reshape, concatenate, \
    Dense, Lambda
from keras_applications.resnet50 import identity_block, conv_block
import tensorflow as tf
from losses import class_loss_cls, class_loss_regr


def stage_2_net(nb_classes, input_tensor, input_target, height=80, width=40):
    """
    自己设计的4倍下采样，简易类VGG基础网络
    :param nb_classes: 11类别数
    :param input_tensor: 图片numpy的X
    :param input_target: 分类、回归的目标Y
    :param height: crops图片高
    :param width: crops图片宽
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
    classification = Convolution2D(filters=height * width * 9 * nb_classes // 2400, kernel_size=3, padding='same')(concat)
    classification = Reshape(target_shape=(-1, nb_classes))(classification)
    classification = Activation(activation='softmax', name='classification')(classification)
    # 接上regr输出层
    bboxes_regression = Convolution2D(filters=height * width * 9 * 4 // 240, kernel_size=3, padding='same')(concat)
    bboxes_regression = Reshape(target_shape=(-1, 4*(nb_classes-1)), name='regression')(bboxes_regression)

    # add loss layer
    # 输入：不变（自变量：X、因变量：Y_true）
    # 输出：Y_pred  ——> loss函数返回值（没有分支）
    # model初始化Model的时候，就把目标参数的shape当作参数传入model，然后训练时再gen_data迭代器（[X, [Y1, Y2]]）?
    # 如：
    #    target_input1 = Input(shape=(5, 2400, 12))
    #    target_input2 = Input(shape=(5, 2400, 80))
    loss_cls = Lambda(class_loss_cls, name='cls_loss')([input_target[0], classification])
    loss_regr = Lambda(class_loss_regr(nb_classes - 1), name='regr_loss')([input_target[1], bboxes_regression])
    '''
    # 接上cls输出层
    classification = Dense(nb_classes, activation='softmax', kernel_initializer='zero')(concat)
    # 接上regr输出层
    bboxes_regression = Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero')(concat)
    '''
    # 最终的model构建
    # detect_model = Model(inputs=[input_tensor, input_target], outputs=[loss_cls, loss_regr])
    # detect_model.summary()

    return [loss_cls, loss_regr]

if __name__ == "__main__":
    target_input1 = Input(shape=(5, 2400, 12))
    target_input2 = Input(shape=(5, 2400, 80))
    stage_2_net(11, Input(shape=(80, 40, 3)), [target_input1, target_input2], height=80, width=40)
