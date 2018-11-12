import random
import pprint
import time
import os
import numpy as np
import pickle
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
from props_pic_2nd import props_pic, pic_num_label
from anchor_2nd import anchors_generation, sliding_anchors_all, pos_neg_iou, anchor_targets_bbox
from net_design_2nd import stage_2_net


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
    # random.shuffle(all_images)

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
    small_img_input = Input(shape=(40, 20, 3))

    # 定义基础网络
    shared_layers = nn.nn_base(img_input, trainable=True)

    # 定义rpn网络
    num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)  # 9
    rpn = nn.rpn(shared_layers, num_anchors)

    # 定义后续分类网络的输出
    classifier = stage_2_net(len(classes_count), small_img_input, height=40, width=20)

    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model(small_img_input, classifier)

    model_all = Model([img_input, small_img_input], rpn[:2] + classifier)

    # 加载预训练模型参数
    print('\n======== 加载预训练模型参数 ========')
    try:
        print('loading weights from {}'.format(cfg.base_net_weights))
        model_rpn.load_weights(cfg.base_model_path, by_name=True)
        model_classifier.load_weights(cfg.base_model_path, by_name=True)
    except Exception as e:
        print(e)
        print('Could not load pretrained model weights. Weights can be found in the keras application folder '
              'https://github.com/fchollet/keras/tree/master/keras/applications')

    # 编译模型
    optimizer = Adam(lr=1e-5)
    optimizer_classifier = Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer,
                      loss=[losses_fn.rpn_loss_cls(num_anchors), losses_fn.rpn_loss_regr(num_anchors)])
    model_classifier.compile(optimizer=optimizer_classifier,
                             loss=[losses_fn.class_loss_cls, losses_fn.class_loss_regr(len(classes_count) - 1)],
                             metrics={'dense_class_{}'.format(len(classes_count)): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

    # 设置一些训练参数
    epoch_length = 100
    num_epochs = 10
    losses = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []
    start_time = time.time()
    best_loss = np.Inf

    iter_num = 0
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
                # 训练rpn
                loss_rpn = model_rpn.train_on_batch(X, Y)

                # 预测每个anchor的分数和回归系数, P_rpn[0]维度为(1,m,n,9), P_rpn[1]维度为(1,m,n,36)
                P_rpn = model_rpn.predict_on_batch(X)
                # 在feature map上生成按预测得分降序排列的proposals（即rois）
                result = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], cfg, K.image_dim_ordering(), use_regr=True,
                                                overlap_thresh=0.7,
                                                max_boxes=10)
                # feature map上的proposals坐标映射回resize原图(16倍下采样)
                result[:, :] = 16 * result[:, :]
                # 计算10个proposals与图片tag的overlap，若返回不为空，则筛选出来并crops出原图对应区域
                x1_tag = img_data['outer_boxes'][0]['x1'] // 1.8 # resize缩放因子=1.8
                y1_tag = img_data['outer_boxes'][0]['y1'] // 1.8
                x2_tag = img_data['outer_boxes'][0]['x2'] // 1.8
                y2_tag = img_data['outer_boxes'][0]['y2'] // 1.8
                cls_tag = img_data['outer_boxes'][0]['class']
                # rs_pic是保留下来的crop图像，rs_wh是每张原图crop出的多个rs_pic，对应的统一长、宽
                rs_pic, rs_boxes, rs_wh = props_pic(result[np.newaxis, :, :], [[x1_tag, y1_tag, x2_tag, y2_tag, cls_tag]], X[np.newaxis, :, :, :])
                # 定义第2阶段crops后的号码牌数字标签坐标与原数字标签坐标之间的关系
                annos_list = [[],[],[],[],[]]
                for i in range(len(img_data['bboxes'])):
                    annos_list[0].append(img_data['bboxes'][i]['x1']//1.8)
                    annos_list[1].append(img_data['bboxes'][i]['y1']//1.8)
                    annos_list[2].append(img_data['bboxes'][i]['x2']//1.8)
                    annos_list[3].append(img_data['bboxes'][i]['y2']//1.8)
                    annos_list[4].append(img_data['bboxes'][i]['class'])
                annos_np = np.concatenate((np.array(annos_list[0])[np.newaxis, :],np.array(annos_list[1])[np.newaxis, :],np.array(annos_list[2])[np.newaxis, :],np.array(annos_list[3])[np.newaxis, :],np.array(annos_list[4])[np.newaxis, :]), axis=0).T
                num_labels = pic_num_label(rs_pic, rs_boxes, rs_wh, annos_np[np.newaxis, :, :])

                # ==============================================================================
                # 生成第二阶段的训练数据
                # ==============================================================================
                batch_size = len(num_labels[0])
                base_anchors = anchors_generation(2.5, [1], [0.5**(1.0/3.0),1,2 ** (1.0/2.0)])
                all_anchors = sliding_anchors_all((10,5), (4, 4), base_anchors)
                labels_batch, regression_batch, boxes_batch, inds, pos_inds = anchor_targets_bbox(all_anchors, rs_pic[0], num_labels[0],
                                                                                  len(class_mapping)-1)
                x1 = rs_pic[0] # tf.tensor转换为numpy
                # Y1 = [labels_batch, regression_batch]

                # 区分训练过程中计算loss的anchors样本，并提取非背景类的anchors索引
                rpn_accuracy_rpn_monitor.append(len(inds[0]))
                rpn_accuracy_for_epoch.append(len(inds[0]))

                # 训练分类网络
                # y1目标数据
                labels_batch[:, inds[0], -1] = np.abs(labels_batch[:, inds[0], -1] - 1)
                y1 = labels_batch[:, inds[0], :]
                print('batch_{}第2阶段正负样本数量：{}'.format(n, len(y1[0])))
                # y2目标数据
                tmp = np.zeros((len(y1[0]), 4 * (len(classes_count) - 1)))
                for i in range(len(y1[0])):
                    a = np.zeros(4 * (len(classes_count) - 1))
                    # print(labels_batch[:, inds[0], -1]) # 没有正样本
                    if labels_batch[:, inds[0], -1][0][i] == 0:
                        label_index = list(labels_batch[:, inds[0], :(len(classes_count)-1)][0][i]).index(1)
                        a[4 * label_index: 4 * label_index + 4] = regression_batch[:, inds[0], :4][0][i]
                    tmp[i] = a
                # 合并为list（3维）
                y2 = np.concatenate([np.repeat(y1[:, :, :(len(classes_count) - 1)], 4, axis=2), tmp[np.newaxis, :, :]], axis=2)
                print(x1.shape)
                print(y1.shape)
                print(y2.shape)
                loss_class = model_classifier.train_on_batch(x1,[y1, y2])

                # 统计loss
                losses[iter_num, 0] = loss_rpn[1]
                losses[iter_num, 1] = loss_rpn[2]
                losses[iter_num, 2] = loss_class[1]
                losses[iter_num, 3] = loss_class[2]
                losses[iter_num, 4] = loss_class[3]

                # 更新进度条
                iter_num += 1
                progbar.update(iter_num,
                               [('rpn_cls', np.mean(losses[:iter_num, 0])), ('rpn_regr', np.mean(losses[:iter_num, 1])),
                                ('detector_cls', np.mean(losses[:iter_num, 2])),
                                ('detector_regr', np.mean(losses[:iter_num, 3]))])
                n += 1
                # 如果一个epoch结束，输出各个部分的平均误差
                if iter_num == epoch_length:
                    loss_rpn_cls = np.mean(losses[:, 0])
                    loss_rpn_regr = np.mean(losses[:, 1])
                    loss_class_cls = np.mean(losses[:, 2])
                    loss_class_regr = np.mean(losses[:, 3])
                    class_acc = np.mean(losses[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    # 输出提示信息
                    if cfg.verbose:
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                            mean_overlapping_bboxes))
                        print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {}'.format(loss_rpn_regr))
                        print('Loss Detector classifier: {}'.format(loss_class_cls))
                        print('Loss Detector regression: {}'.format(loss_class_regr))
                        print('Elapsed time: {}'.format(time.time() - start_time))

                    # 当前整个epoch的总和损失
                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

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
print('Training complete, exiting.')


if __name__ == '__main__':
    train()
