# -*- coding: utf-8 -*-
"""
Created on 2018/11/19 10:00

@author: royce.mao

# 把loss写进layer之后，2阶段网络结构的合并
"""
import os
import cv2
import time
import numpy as np
from keras.utils import generic_utils
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras import backend as K
from keras.preprocessing.image import img_to_array
import roi_helpers
import config, pprint, pickle, random
import resnet_rpn_loss_layer as nn
from simple_parser import get_data
from net_design_loss_layer import stage_2_net_res

def data_gen_train_merge():
    return 0


def model_all(nb_classes, cfg, input_tensor, target_1st, target_2nd, height=160, width=80):
    """
    把loss写成layer融进去网络之后，前后2阶段网络的端到端对接
    :param nb_classes: 
    :param input_tensor: 
    :param input_target: 
    :param height: 
    :param width: 
    :return: 
    """
    # 定义原1阶段的基础网络
    shared_layers = nn.nn_base(input_tensor, trainable=True) # resnet50的基础特征提取网络
    num_anchors = 9  # 单个锚点的anchors生成数量(源码固定为9)
    [cls_layer, regr_layer] = nn.rpn(shared_layers, num_anchors, target_1st) # rpn网络分支输出

    # 联合部分：
    ## proposals回归
    proposals = roi_helpers.rpn_to_roi(cls_layer, regr_layer, cfg, K.image_dim_ordering(), use_regr=True,
                                                overlap_thresh=0.7,
                                                max_boxes=5)
    ## 坐标映射回原图
    proposals[:, :] = 16 * proposals[:, :]
    ## 接上input_tensor原图，在上面做Top-5的proposals的crop和resize，得到下一阶段的输入
    width_mean = 80
    height_mean = 160
    input_tensor_cls = []
    for j, prop in enumerate(proposals):
        prop_crop = input_tensor.crop([prop[0]-7, prop[1]-15, prop[2]+7, prop[3]+15])
        prop_crop = img_to_array(prop_crop)
        prop_pic = cv2.resize(prop_crop, (width_mean, height_mean))  # 图片标准化resize
        input_tensor_cls.append(prop_pic)
    ##
    # 定义原2阶段的基础网络
    classification, bboxes_regression = stage_2_net_res(nb_classes, np.array(input_tensor_cls), target_2nd, height=160, width=80)
    return [cls_layer, regr_layer], [classification, bboxes_regression]


def train():
    cfg = config.Config()
    # 基础数据读取与参数
    cfg.use_horizontal_flips = True
    cfg.use_vertical_flips = True
    cfg.rot_90 = True
    cfg.base_net_weights = os.path.join('./model/', nn.get_weight_path())
    # cfg.model_path = './model/kitti_frcnn_last.hdf5'
    cfg.simple_label_file = 'img_infos.txt'
    # 读取VOC图片数据
    print('======== 读取图片信息 ========')
    all_images, classes_count, class_mapping = get_data(cfg.simple_label_file)
    # 增加背景类（若没有）
    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)
    cfg.class_mapping = class_mapping
    print('class_count:')
    pprint.pprint(classes_count)
    print('类别数量：{}'.format(len(classes_count)))
    # # 打乱图片顺序
    random.shuffle(all_images)
    # 分配训练集和测试集
    train_imgs = [s for s in all_images if s['imageset'] == 'trainval']
    val_imgs = [s for s in all_images if s['imageset'] == 'test']
    print('训练集数量：{}'.format(len(train_imgs)))
    print('测试集数量：{}'.format(len(val_imgs)))
    # 保存配置文件
    print('\n======== 保存配置 ========')
    with open(cfg.config_save_file, 'wb') as config_f:
        pickle.dump(cfg, config_f)
        print('配置参数已写入：{},'.format(cfg.config_save_file))

    # 定义输入网络的数据维度及部分参数
    img_input = Input(shape=(None, None, 3))
    small_img_input = Input(shape=(160, 80, 3)) # 高为80，宽为40
    nb_classes = len(classes_count)
    # 定义输入网络的目标Y_true


    # 定义后续分类网络的输出
    rpn_out, classifier_out = model_all(nb_classes, cfg, img_input, target_1st, target_2nd, height=160, width=80)
    model_merge = Model([img_input, small_img_input], rpn_out + classifier_out)

    # 加载预训练模型参数
    print('\n======== 加载预训练模型参数 ========')
    try:
        print('loading weights from {}'.format(cfg.base_net_weights))
        model_merge.load_weights(cfg.base_model_path, by_name=True)
    except Exception as e:
        print(e)
        print('Could not load pretrained model weights. Weights can be found in the keras application folder '
          'https://github.com/fchollet/keras/tree/master/keras/applications')

    # 编译模型
    optimizer = Adam(lr=1e-4)
    model_merge.compile(optimizer=optimizer,
                        loss=None,
                        metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})

    # 开始训练
    ## 设置一些训练参数
    epoch_length = 120
    num_epochs = 3
    losses = np.zeros((epoch_length, 5))
    start_time = time.time()
    best_loss = np.Inf
    iter_num = 0
    print('\n======== 开始训练 ========')
    for epoch_num in range(num_epochs):
        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
        n = 1
        while True:
            X, Y, img_data = next(data_gen_train_merge)
            model_merge.train_on_batch(X, Y)