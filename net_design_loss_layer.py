# -*- coding: utf-8 -*-
"""
Created on 2018/11/7 10:34
@author: royce.mao
# 构造第2阶段，小图片数字的识别检测网络
"""
from __future__ import print_function
from __future__ import absolute_import

from keras.models import Model
from keras.layers import Conv2D, Reshape, Lambda, Activation, Convolution2D, MaxPooling2D, ZeroPadding2D, Add, BatchNormalization, concatenate, Dense
from fixed_batch_normalization import FixedBatchNormalization
from keras import backend as K
from losses_loss_layer import class_loss_cls, class_loss_regr

def stage_2_net(nb_classes, input_tensor, height=160, width=80):
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
    classification = Convolution2D(filters=height * width * 12 * nb_classes // 12800, kernel_size=3, padding='same')(concat)
    classification = Reshape(target_shape=(-1, nb_classes))(classification)
    classification = Activation(activation='softmax', name='classification')(classification)
    # 接上regr输出层
    bboxes_regression = Convolution2D(filters=height * width * 12 * 4 // 1280, kernel_size=3, padding='same')(concat)
    bboxes_regression = Reshape(target_shape=(-1, 4*(nb_classes-1)), name='regression')(bboxes_regression)
    '''
    # 接上cls输出层
    classification = Dense(nb_classes, activation='softmax', kernel_initializer='zero')(concat)
    # 接上regr输出层
    bboxes_regression = Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero')(concat)
    '''
    # 最终的model构建
    detect_model = Model(inputs=input_tensor, outputs=[classification, bboxes_regression])
    detect_model.summary()

    return [classification, bboxes_regression]


def stage_2_net_vgg(nb_classes, input_tensor, height = 160, width = 80):
    """
    VGG网络的前3个blocks，8倍下采样
    :param input_tensor: 
    :param trainable: 
    :return: 
    """
    bn_axis = 3

    # Block 1
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # # 接上cls输出层
    # classification = Dense(nb_classes, activation='softmax', kernel_initializer='zero')(x)
    # # 接上regr输出层
    # bboxes_regression = Dense(4 * (nb_classes - 1), activation='linear', kernel_initializer='zero')(x)

    # 接上cls输出层
    classification = Convolution2D(filters=height * width * 18 * nb_classes // 12800, kernel_size=3, padding='same')(x)
    classification = Reshape(target_shape=(-1, nb_classes))(classification)
    classification = Activation(activation='softmax', name='classification')(classification)
    # 接上regr输出层
    bboxes_regression = Convolution2D(filters=height * width * 18 * 4 // 1280, kernel_size=3, padding='same')(x)
    bboxes_regression = Reshape(target_shape=(-1, 4*(nb_classes-1)), name='regression')(bboxes_regression)

    # detect_model = Model(inputs=input_tensor, outputs=[classification, bboxes_regression])
    # detect_model.summary()
    return [classification, bboxes_regression]

def stage_2_net_res(nb_classes, input_tensor, input_target_cls, input_target_regr, height = 160, width = 80):
    """
    resnet网络的前2个blocks，8倍下采样
    :param input_tensor: 
    :param trainable: 
    :return: 
    """
    bn_axis = 3
    x = ZeroPadding2D((3, 3))(input_tensor)

    x = Convolution2D(64, (7, 7), strides=(2, 2), name='conv1')(x)

    # NOTE: this code only support to keras 2.0.3, newest version this line will got errors.
    x = FixedBatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # 接上cls输出层
    classification = Convolution2D(filters=height * width * 18 * nb_classes // 12800, kernel_size=3, padding='same')(x)
    classification = Reshape(target_shape=(-1, nb_classes))(classification)
    classification = Activation(activation='softmax', name='classification')(classification)
    # 接上regr输出层
    bboxes_regression = Convolution2D(filters=height * width * 18 * 4 // 1280, kernel_size=3, padding='same')(x)
    bboxes_regression = Reshape(target_shape=(-1, 4*(nb_classes-1)), name='regression')(bboxes_regression)

    # add loss layer
    # 输入：不变（自变量：X、因变量：Y_true）
    # 输出：Y_pred  ——> loss函数返回值（没有分支）
    # model初始化Model的时候，就把目标参数的shape当作参数传入model，然后训练时再gen_data迭代器（[X, [Y1, Y2]]）?
    # 如：
    #    target_input1 = Input(shape=(5, 2400, 12))
    #    target_input2 = Input(shape=(5, 2400, 80))
    loss_cls = Lambda(lambda x: class_loss_cls(*x), name='cls_loss')([input_target_cls, classification])
    loss_regr = Lambda(lambda x: class_loss_regr(*x), name='regr_loss')([input_target_regr, bboxes_regression])

    # detect_model = Model(inputs=input_tensor, outputs=[loss_cls, loss_regr])
    # detect_model.summary()

    return [loss_cls, loss_regr], [classification, bboxes_regression]

def identity_block(input_tensor, kernel_size, filters, stage, block, trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters

    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), name=conv_name_base + '2a', trainable=trainable)(input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b',
                      trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
    nb_filter1, nb_filter2, nb_filter3 = filters
    if K.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, (1, 1), strides=strides, name=conv_name_base + '2a', trainable=trainable)(
        input_tensor)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b',
                      trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, (1, 1), name=conv_name_base + '2c', trainable=trainable)(x)
    x = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, (1, 1), strides=strides, name=conv_name_base + '1', trainable=trainable)(
        input_tensor)
    shortcut = FixedBatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x
