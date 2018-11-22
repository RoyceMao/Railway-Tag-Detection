# -*- coding: utf-8 -*-
"""VGG16 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
"""
from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input, Conv2D, Convolution2D, MaxPooling2D, Activation, BatchNormalization, Reshape, concatenate
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D, TimeDistributed
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras import backend as K
from roi_pooling_conv import RoiPoolingConv


def stage_2_net(nb_classes, input_tensor, height = 160, width = 80):
    """
    VGG网络的前2个blocks，并加载预训练模型
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
    '''
    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    '''
    # 接上cls输出层
    classification = Convolution2D(filters=height * width * 12 * nb_classes // 12800, kernel_size=3, padding='same')(x)
    classification = Reshape(target_shape=(-1, nb_classes))(classification)
    classification = Activation(activation='softmax', name='classification')(classification)
    # 接上regr输出层
    bboxes_regression = Convolution2D(filters=height * width * 12 * 4 // 1280, kernel_size=3, padding='same')(x)
    bboxes_regression = Reshape(target_shape=(-1, 4*(nb_classes-1)), name='regression')(bboxes_regression)

    # detect_model = Model(inputs=input_tensor, outputs=[classification, bboxes_regression])
    # detect_model.summary()
    return [classification, bboxes_regression]

if __name__ == "__main__":
    stage_2_net(11, Input(shape=(160, 80, 3)), height=160, width=80)
