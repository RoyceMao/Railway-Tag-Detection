# -*- coding: utf-8 -*-
"""
Created on 2018/11/19 10:00

@author: royce.mao

# 把loss写进layer之后，2阶段网络结构的合并
"""
import os
import time
import numpy as np
from keras.utils import generic_utils
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
from keras import backend as K
import data_generators
from props_pic_2nd import props_pic
import roi_helpers
import config, pprint, pickle, random
import resnet_rpn_loss_layer as nn
from simple_parser import get_data
from net_design_loss_layer import stage_2_net_res
from anchor_2nd import anchors_generation, sliding_anchors_all, pos_neg_iou, anchor_targets_bbox
np.seterr(divide='ignore', invalid='ignore') # 不允许“divide”Warning相关信息的打印


def data_gen_stage_2(result, img_data, X, class_mapping, classes_count, iter_num):
    """
    根据1阶段的一个batch（1张）图片处理结果，生成第2阶段的训练数据
    :param result: 
    :param img_data: 
    :param sess: 
    :param X: 
    :param class_mapping: 
    :return: 
    """
    # feature map上的proposals坐标映射回resize原图(16倍下采样)
    result[:, :] = 16 * result[:, :]
    # 提取第1阶段每个batch图片对应的Tag标签标注
    x1_tag = img_data['outer_boxes'][0]['x1']
    y1_tag = img_data['outer_boxes'][0]['y1']
    x2_tag = img_data['outer_boxes'][0]['x2']
    y2_tag = img_data['outer_boxes'][0]['y2']
    cls_tag = img_data['outer_boxes'][0]['class']
    # 提取第1阶段每个batch图片对应的数字标签标注
    annos_list = [[], [], [], [], []]
    for i in range(len(img_data['bboxes'])):
        # 对无法辨别的小数字标注（高、宽均低于原图的10个像素点）做剔除，防止影响最终检测效果
        if img_data['bboxes'][i]['x2'] - img_data['bboxes'][i]['x1'] >= 10 and img_data['bboxes'][i]['y2'] - \
                img_data['bboxes'][i]['y1'] >= 10:
            annos_list[0].append(img_data['bboxes'][i]['x1'])
            annos_list[1].append(img_data['bboxes'][i]['y1'])
            annos_list[2].append(img_data['bboxes'][i]['x2'])
            annos_list[3].append(img_data['bboxes'][i]['y2'])
            annos_list[4].append(img_data['bboxes'][i]['class'])
    annos_np = np.concatenate((np.array(annos_list[0])[np.newaxis, :], np.array(annos_list[1])[np.newaxis, :],
                               np.array(annos_list[2])[np.newaxis, :], np.array(annos_list[3])[np.newaxis, :],
                               np.array(annos_list[4])[np.newaxis, :]), axis=0).T
    # 进行第2阶段所需参数的计算提取过程
    rs_pic, rs_boxes, rs_num_gt_pic, rs_wh, gt_index = props_pic(result[np.newaxis, :, :],
                                                                 [[x1_tag, y1_tag, x2_tag, y2_tag, cls_tag]],
                                                                 annos_np[np.newaxis, :, :], X[np.newaxis, :, :, :])
    # ==============================================================================
    # 生成第二阶段的训练数据
    # ==============================================================================
    batch_size = len(rs_pic[0])  # 一次5张crops图
    base_anchors = anchors_generation(16, [0.5 ** (1.0 / 3.0), 1, 2 ** (1.0 / 3.0)],
                                      [0.5, 0.5 ** (1.0 / 2.0), 1, 2 ** (1.0 / 3.0), 2 ** (1.0 / 2.0), 2])
    all_anchors = sliding_anchors_all((10, 20), (8, 8), base_anchors)
    # ================================================================
    '''
    # 测试部分：计算当前anchor与当前gt-box的iou，以及框住gt的proposals生成的anchors是否覆盖了所有的小gt
    from overlap_2nd import overlap
    if gt_index != [[]]:
        for gt in rs_num_gt_pic[0][gt_index[0][0]]:
            num_width.append(gt[2]-gt[0])
            num_height.append(gt[3]-gt[1])

        # 打印Top-5的最高IOU值
        for anchor in all_anchors:
            for gt in  rs_num_gt_pic[0][gt_index[0][0]]:
                if overlap(anchor[np.newaxis, :], gt[np.newaxis, :])[0][0] > max:
                    max = overlap(anchor[np.newaxis, :], gt[np.newaxis, :])[0][0]
        print("\nTop-{}最高IOU值：{}\n".format(gt_index[0][0]+1,max))
    '''

    # print("\n宽均值：{}、高均值：{}、宽_max：{}、高_max：{}、宽_min：{}、高_min：{}、高宽比_max：{}、高宽比_min：{}".format(mean_width,mean_height,max_width,max_height,min_width,min_height,max_ratio,min_ratio))
    # print(np.array(rs_num_gt_pic[0]))
    # ================================================================
    labels_batch, regression_batch, boxes_batch, inds, pos_inds = anchor_targets_bbox(all_anchors, rs_pic[0],
                                                                                      rs_num_gt_pic[0],
                                                                                      len(class_mapping) - 1)
    # ============================================================
    # ============================================================
    x1 = rs_pic[0]  # tf.tensor转换为numpy
    # Y1 = [labels_batch, regression_batch]

    # 区分训练过程中计算loss的anchors样本，并提取非背景类的anchors索引
    # ===========rpn_accuracy_rpn_monitor.append(len(inds[0]))
    # ===========rpn_accuracy_for_epoch.append(len(inds[0]))
    # 训练分类网络
    # y1目标数据
    labels_batch[inds, -1] = np.abs(labels_batch[inds, -1] - 1)  # 0、1标注减1再取绝对值，相当于把labels_batch第3维最后1列由前景背景类替换为bg类
    # labels_batch[:, :, -1] = np.abs(labels_batch[:, :, -1]) # 然后所有最后1列取绝对值把中性样本也替换为bg类
    # 正、负样本与中性样本的区分numpy
    tmp = np.zeros((batch_size, len(inds[0]), 1))
    for batch in range(len(inds)):
        for i in range(len(inds[0])):
            a = np.zeros(1)
            # print(labels_batch[:, inds[0], -1])
            if labels_batch[:, :, -1][batch][i] != -1:  # 说明是非忽略样本
                a = 1
            tmp[batch][i] = a
    y1 = np.concatenate([tmp, labels_batch], axis=2)
    # y2目标数据
    tmp = np.zeros((batch_size, len(inds[0]), 4 * (len(classes_count) - 1)))
    for batch in range(len(inds)):
        for i in range(len(inds[0])):
            a = np.zeros(4 * (len(classes_count) - 1))
            # print(labels_batch[:, inds[0], -1])
            if labels_batch[:, :, -1][batch][i] == 0:  # 说明是正样本
                # 取出正样本赋值标签为1的类别索引
                label_index = list(labels_batch[:, :, :(len(classes_count) - 1)][batch][i]).index(1)
                # 然后把对应的正样本回归目标进行对应类别下的赋值
                a[4 * label_index: 4 * label_index + 4] = regression_batch[:, :, :4][batch][i]
            tmp[batch][i] = a

    # 合并为list
    y2 = np.concatenate([np.repeat(labels_batch[:, :, :(len(classes_count) - 1)], 4, axis=2), tmp], axis=2)
    return np.array(x1), [y1, y2]


def model_all(nb_classes, cfg, height=160, width=80):
    """
    把loss写成layer融进去网络之后，前后2阶段网络的端到端对接
    :param nb_classes: 
    :param input_tensor: 
    :param input_target: 
    :param height: 
    :param width: 
    :return: 
    """
    # 定义输入网络的pic数据维度（跟设置有关）
    input_tensor = Input(shape=(None, None, 3))
    small_img_input = Input(shape=(160, 80, 3))
    # 定义输入网络的目标Y_true维度（跟data_gen迭代器函数有关）
    target_1st_cls = Input(shape=(68, 120, 18))
    target_1st_regr = Input(shape=(68, 120, 72))
    target_2nd_cls = Input(shape=(3600, 12))
    target_2nd_regr = Input(shape=(3600, 80))
    # 定义原1阶段的基础网络
    shared_layers = nn.nn_base(input_tensor, trainable=True) # resnet50的基础特征提取网络
    num_anchors = 9  # 单个锚点的anchors生成数量(源码固定为9)
    [x_class, x_regr], [x_loss_cls, x_loss_regr] = nn.rpn(shared_layers, num_anchors, target_1st_cls, target_1st_regr) # rpn网络分支输出
    '''
    # 联合部分：
    ## proposals回归（这部分封装为layer时，[cls_layer, regr_layer]两个参数都是Tensor，不是numpy，如何做回归？）
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
    '''
    # 定义原2阶段的基础网络
    [loss_cls, loss_regr], [classification, bboxes_regression] = stage_2_net_res(nb_classes,  small_img_input, target_2nd_cls, target_2nd_regr, height=160, width=80)
    model_rpn = Model([input_tensor, target_1st_cls, target_1st_regr], [x_class, x_regr])
    model_cls = Model([small_img_input, target_2nd_cls, target_2nd_regr], [classification, bboxes_regression])
    model_merge = Model([input_tensor, small_img_input, target_1st_cls, target_1st_regr, target_2nd_cls, target_2nd_regr], [classification, bboxes_regression])
    return model_rpn, model_cls, model_merge, [x_loss_cls, x_loss_regr], [loss_cls, loss_regr]


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
    # 增加背景类
    if 'bg' not in classes_count:
        classes_count['bg'] = 0
        class_mapping['bg'] = len(class_mapping)
    cfg.class_mapping = class_mapping
    print('class_count:')
    pprint.pprint(classes_count)
    print('类别数量：{}'.format(len(classes_count)))
    # # 打乱图片顺序
    # random.shuffle(all_images)
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

    # 返回数据生成器
    data_gen_train = data_generators.get_anchor_gt(train_imgs, cfg, nn.get_img_output_length,
                                                   K.image_dim_ordering(), mode='train')
    # 初始化model，并add_loss_layer
    nb_classes = len(classes_count)
    model_rpn, model_cls, model_merge, [x_loss_cls, x_loss_regr], [loss_cls, loss_regr] = model_all(nb_classes, cfg, height=160, width=80)
    model_rpn.add_loss([x_loss_cls, x_loss_regr])
    model_cls.add_loss([loss_cls, loss_regr])

    # 加载预训练模型参数
    print('\n======== 加载预训练模型参数 ========')
    try:
        print('loading weights from {}'.format(cfg.base_net_weights))
        model_rpn.load_weights(cfg.base_rpn_model_path, by_name=True)
        model_cls.load_weights(cfg.base_tf_model_path, by_name=True)
    except Exception as e:
        print(e)
        print('Could not load pretrained model weights. Weights can be found in the keras application folder '
          'https://github.com/fchollet/keras/tree/master/keras/applications')

    # 编译模型
    optimizer = Adam(lr=1e-5)
    optimizer_classifier = Adam(lr=1e-3)
    model_rpn.compile(optimizer=optimizer,
                      loss=[None] * 2)
    model_cls.compile(optimizer=optimizer_classifier,
                      loss=[None] * 2)
    model_merge.compile(optimizer='sgd', loss='mae')

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
            X, Y, img_data, X_2  = next(data_gen_train)
            loss_rpn = model_rpn.train_on_batch([X,Y[0],Y[1]], None)
            P_rpn = model_rpn.predict_on_batch([X,Y[0],Y[1]])
            # 在feature map上生成按预测得分降序排列的proposals（即rois）
            result = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], cfg, K.image_dim_ordering(), use_regr=True,
                                            overlap_thresh=0.7,
                                            max_boxes=5)
            # 训练2阶段的classifier
            x, y = data_gen_stage_2(result, img_data, X_2, class_mapping, classes_count, iter_num)
            loss_class = model_cls.train_on_batch([x,y[0],y[1]], None)
            # 统计loss
            losses[iter_num, 0] = loss_rpn
            losses[iter_num, 1] = loss_class

            # 更新进度条   #========== ('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
            iter_num += 1
            progbar.update(iter_num,
                           [('rpn', np.mean(losses[:iter_num, 0])), ('detector', np.mean(losses[:iter_num, 1]))])
            n += 1
            # 如果一个epoch结束，输出各个部分的平均误差
            if iter_num == epoch_length:
                loss_rpn = np.mean(losses[:, 0])
                loss_class = np.mean(losses[:, 1])

                # =======mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                # =======rpn_accuracy_for_epoch = []

                # 输出提示信息
                if cfg.verbose:
                    # ==========print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                    # ===========mean_overlapping_bboxes))
                    print('Loss RPN: {}'.format(loss_rpn))
                    print('Loss Detector: {}'.format(loss_class))
                    print('Elapsed time: {}'.format(time.time() - start_time))

                # 当前整个epoch的总和损失  #=============loss_rpn_cls + loss_rpn_regr +
                curr_loss = loss_rpn + loss_class
                iter_num = 0
                start_time = time.time()

                # 如果当前损失最小，则保存当前的参数
                if curr_loss < best_loss:
                    if cfg.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                    best_loss = curr_loss
                    model_merge.save_weights(cfg.model_cls_path)

                break


if __name__ == '__main__':
    train()
    print('Training complete, exiting.')