# -*- coding: utf-8 -*-
"""
Created on 2018/11/19 10:00

@author: royce.mao

# 把loss写进layer之后，2阶段网络结构的合并
"""
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
import os
import config, pprint, pickle, random
import resnet as nn
from simple_parser import get_data
from net_design_loss_layer import stage_2_net

def model_all(nb_classes, input_tensor, target_1st, target_2nd, height=80, width=40):
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
    num_anchors = 9  # 单个锚点的anchors生成数量
    rpn_loss = nn.rpn(shared_layers, num_anchors, target_1st) # rpn网络分支输出

    # 联合部分：难点


    # 定义原2阶段的基础网络
    classifier_loss = stage_2_net(nb_classes, Input(shape=(10, 20, 3)), target_2nd, height=80, width=40)
    return classifier_loss


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

    # 定义输入网络的数据维度
    img_input = Input(shape=(None, None, 3))
    x_target = Input(shape=(None, 38, 67, 18))
    y_target = Input(shape=(None, 38, 67, 72))
    target_input1 = Input(shape=(None, 2400, 12))
    target_input2 = Input(shape=(None, 2400, 80))

    # 定义后续分类网络的输出
    loss = model_all(len(classes_count), img_input, [x_target, y_target], [target_input1, target_input2], height=80, width=40)
    model_merge = Model([img_input, [x_target, y_target], [target_input1, target_input2]], loss)

    # 加载预训练模型参数
    print('\n======== 加载预训练模型参数 ========')
    try:
        print('loading weights from {}'.format(cfg.base_net_weights))
        model_merge.load_weights(cfg.base_rpn_model_path, by_name=True)
    except Exception as e:
        print(e)
        print('Could not load pretrained model weights. Weights can be found in the keras application folder '
          'https://github.com/fchollet/keras/tree/master/keras/applications')

    # 编译模型
    optimizer = Adam(lr=1e-5)
    model_merge.compile(optimizer=optimizer,
                        loss=None,
                        metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})

