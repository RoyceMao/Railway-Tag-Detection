import random
import pprint
import time
import os
import gc
import cv2
import numpy as np
import pickle
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model
import config
import data_generators
import losses as losses_fn
import roi_helpers as roi_helpers
from keras.utils import generic_utils
import resnet as nn
from simple_parser import get_data
from props_pic_2nd import props_pic
from Visual import _create_unique_color_float, _create_unique_color_uchar, draw_boxes_and_label_on_image
from anchor_2nd import anchors_generation, sliding_anchors_all, pos_neg_iou, anchor_targets_bbox
from net_design_2nd import stage_2_net
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)


def data_gen_stage_2(result, img_data, sess, X, class_mapping, classes_count, num_width, num_height, iter_num):
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
    x1_tag = img_data['outer_boxes'][0]['x1'] // 1.8  # resize缩放因子=1.8
    y1_tag = img_data['outer_boxes'][0]['y1'] // 1.8
    x2_tag = img_data['outer_boxes'][0]['x2'] // 1.8
    y2_tag = img_data['outer_boxes'][0]['y2'] // 1.8
    cls_tag = img_data['outer_boxes'][0]['class']
    # 提取第1阶段每个batch图片对应的数字标签标注
    annos_list = [[], [], [], [], []]
    for i in range(len(img_data['bboxes'])):
        # 对无法辨别的小数字标注（高、宽均低于原图的10个像素点）做剔除，防止影响最终检测效果
        if img_data['bboxes'][i]['x2'] - img_data['bboxes'][i]['x1'] >= 10 and img_data['bboxes'][i]['y2'] - \
                img_data['bboxes'][i]['y1'] >= 10:
            annos_list[0].append(img_data['bboxes'][i]['x1'] // 1.8)
            annos_list[1].append(img_data['bboxes'][i]['y1'] // 1.8)
            annos_list[2].append(img_data['bboxes'][i]['x2'] // 1.8)
            annos_list[3].append(img_data['bboxes'][i]['y2'] // 1.8)
            annos_list[4].append(img_data['bboxes'][i]['class'])
    annos_np = np.concatenate((np.array(annos_list[0])[np.newaxis, :], np.array(annos_list[1])[np.newaxis, :],
                               np.array(annos_list[2])[np.newaxis, :], np.array(annos_list[3])[np.newaxis, :],
                               np.array(annos_list[4])[np.newaxis, :]), axis=0).T
    # 进行第2阶段所需参数的计算提取过程
    rs_pic, rs_boxes, rs_num_gt_pic, rs_wh, gt_index = props_pic(sess, result[np.newaxis, :, :],
                                                                 [[x1_tag, y1_tag, x2_tag, y2_tag, cls_tag]],
                                                                 annos_np[np.newaxis, :, :], X[np.newaxis, :, :, :])
    # ==============================================================================
    # 生成第二阶段的训练数据
    # ==============================================================================
    batch_size = len(rs_pic[0])  # 一次5张crops图
    base_anchors = anchors_generation(8, [0.5 ** (1.0 / 3.0), 1, 2 ** (1.0 / 2.0)],
                                      [1,2 ** (1.0/3.0),2 ** (1.0/2.0),2])
    all_anchors = sliding_anchors_all((20, 10), (4, 4), base_anchors)
    #================================================================
    # 测试部分：计算当前anchor与当前gt-box的iou，以及框住gt的proposals生成的anchors是否覆盖了所有的小gt
    from overlap_2nd import overlap
    if gt_index != [[]]:
        for gt in rs_num_gt_pic[0][gt_index[0][0]]:
            num_width.append(gt[2]-gt[0])
            num_height.append(gt[3]-gt[1])
    '''
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

    for i, num_gt in enumerate(rs_num_gt_pic[0]):
        if i in gt_index[0]:
            draw_imgs = draw_boxes_and_label_on_image(rs_pic[0][i],
                                                      {1: all_anchors[inds[i]]}) # , {1: num_gt}
            cv2.imwrite('./output_test/Pic{}_Prop{}.png'.format(iter_num, i),
                        draw_imgs)
    #============================================================
    #============================================================
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
    del x1_tag,y1_tag,x2_tag,y2_tag,cls_tag,annos_list,annos_np,rs_pic,rs_boxes,rs_num_gt_pic,rs_wh,gt_index,labels_batch,regression_batch,boxes_batch,inds,pos_inds,tmp
    gc.collect()
    return np.array(x1), [y1,y2], num_width, num_height


def train():
    cfg = config.Config()

    # 下面3行设置了数据增强时所需相关参数
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

    # 分配训练集和测试
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
    # data_gen_val = data_generators.get_anchor_gt(val_imgs, cfg, nn.get_img_output_length,
    #                                              K.image_dim_ordering(), mode='val')

    if K.image_dim_ordering() == 'th':
        input_shape_img = (3, None, None)
    else:
        input_shape_img = (None, None, 3)

    img_input = Input(shape=input_shape_img)
    small_img_input = Input(shape=(80, 40, 3))

    # 定义基础网络
    shared_layers = nn.nn_base(img_input, trainable=True)

    # 定义rpn网络
    num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)  # 9
    rpn = nn.rpn(shared_layers, num_anchors)

    # 定义后续分类网络的输出
    classifier = stage_2_net(len(classes_count), small_img_input, height=80, width=40)

    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model(small_img_input, classifier)

    model_all = Model([img_input, small_img_input], rpn[:2] + classifier)

    # 加载预训练模型参数
    print('\n======== 加载预训练模型参数 ========')
    try:
        print('loading weights from {}'.format(cfg.base_net_weights))
        model_rpn.load_weights(cfg.base_rpn_model_path, by_name=True)
        model_classifier.load_weights(cfg.base_tf_model_path, by_name=True)
    except Exception as e:
        print(e)
        print('Could not load pretrained model weights. Weights can be found in the keras application folder '
              'https://github.com/fchollet/keras/tree/master/keras/applications')

    # 编译模型
    optimizer = Adam(lr=1e-5)
    optimizer_classifier = Adam(lr=1e-4)
    model_rpn.compile(optimizer=optimizer,
                      loss=[losses_fn.rpn_loss_cls(num_anchors), losses_fn.rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier,
                             loss=[losses_fn.class_loss_cls, losses_fn.class_loss_regr(len(classes_count) - 1)],
                             metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    # 设置一些训练参数
    epoch_length = 900
    num_epochs = 1
    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()
    best_loss = np.Inf
    iter_num = 0
    sess = tf.Session()

    #================测试参数
    num_width = []
    num_height = []

    print('\n======== 开始训练 ========')
    for epoch_num in range(num_epochs):
        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(epoch_num + 1, num_epochs))
        n = 1
        while True:
            #try:
                # 当完成一轮epoch时，计算epoch_length个rpn_accuracy的均值，输出相关信息，如果均值为0，则提示出错
                if len(rpn_accuracy_rpn_monitor) == epoch_length and cfg.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
                    print(
                        'Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(
                            mean_overlapping_bboxes, epoch_length))
                    if mean_overlapping_bboxes == 0:
                        print('RPN is not producing bounding boxes that overlap'
                              ' the ground truth boxes. Check RPN settings or keep training.')

                # X：resize后的图片  Y：标定好的anchor和回归系数  img_data：原始图片的信息
                X, Y, img_data = next(data_gen_train)
                # 训练1阶段的rpn
                # loss_rpn = model_rpn.train_on_batch(X, Y)
                # 预测每个anchor的分数和回归系数, P_rpn[0]维度为(1,m,n,9), P_rpn[1]维度为(1,m,n,36)
                P_rpn = model_rpn.predict_on_batch(X)
                # 在feature map上生成按预测得分降序排列的proposals（即rois）
                result = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], cfg, K.image_dim_ordering(), use_regr=True,
                                                overlap_thresh=0.7,
                                                max_boxes=5)
                # 训练2阶段的classifier
                x, y, num_width, num_height = data_gen_stage_2(result, img_data, sess, X, class_mapping, classes_count, num_width, num_height, iter_num)
                loss_class = model_classifier.train_on_batch(x, y)
                # 删除中间参数，释放内存
                del X, Y, img_data, P_rpn, result, x, y
                gc.collect()
                # 统计loss
                #========== losses[iter_num, 0] = loss_rpn[1]
                #========== losses[iter_num, 1] = loss_rpn[2]
                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]

                # 更新进度条   #========== ('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                iter_num += 1
                progbar.update(iter_num,
                               [('detector_cls', np.mean(losses[:iter_num, 2])),('detector_regr', np.mean(losses[:iter_num, 3]))])
                n += 1
                # 如果一个epoch结束，输出各个部分的平均误差
                if iter_num == epoch_length:
                    #======  loss_rpn_cls = np.mean(losses[:, 0])
                    #======  loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])

                    #======mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    #======rpn_accuracy_for_epoch = []

                    # 输出提示信息
                    if cfg.verbose:
                        #=======print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                            #========mean_overlapping_bboxes))
                        #======= print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        #======= print('Loss RPN regression: {}'.format(loss_rpn_regr))
                        print('Loss Detector classifier: {}'.format(loss_class_cls))
                        print('Loss Detector regression: {}'.format(loss_class_regr))
                        print('Elapsed time: {}'.format(time.time() - start_time))

                    # 当前整个epoch的总和损失  #=============loss_rpn_cls + loss_rpn_regr +
                    curr_loss = loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

                    # 1个epoch所有图片proposals区域，正样本anchors的高、宽趋势可视化
                    x = range(len(num_width))
                    y1 = num_width
                    y2 = num_height
                    plt.plot(x, y1, marker='o', color='r')
                    plt.plot(x, y2, marker='*', color='b')
                    # plt.show()
                    # plt.savefig('plot.png',format='png')
                    # 如果当前损失最小，则保存当前的参数
                    if curr_loss < best_loss:
                        if cfg.verbose:
                            print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                        best_loss = curr_loss
                        model_all.save_weights(cfg.model_path)

                    break
'''
            except Exception as e:
                print('Exception: {}'.format(e))
                # save model
                model_all.save_weights(cfg.model_path)
                continue
'''


if __name__ == '__main__':
    train()
    print('Training complete, exiting.')
