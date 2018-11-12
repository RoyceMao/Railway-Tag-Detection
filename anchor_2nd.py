# -*- coding: utf-8 -*-
"""
Created on 2018/10/17 15:30

@author: royce.mao

根据IOU计算生成anchors的过程，按既定比例生成正、负样本anchors

# np.tile：复制数组本身
# np.repeat：复制数组元素
# np.meshgrid：数组行、列复制
# np.vstack：数组堆叠
# np.ravel()：高维数组打平为一维
# np.stack：数组里面的元素堆叠
"""
import keras
from overlap_2nd import overlap
import numpy as np

def anchors_generation(base_size=None, ratios=None, scales=None):
    """
    生成基准anchors
    :param base_size: 基准尺寸大小
    :param ratios: 长宽比
    :param scales: 尺寸缩放列表
    :return: anchor的基准坐标(x1, y1, x2, y2)，相对于中心点(0,0)
    """
    if base_size is None:
        base_size = 18

    if ratios is None:
        ratios = np.array([0.5, 1, 2])

    if scales is None:
        scales = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)])

    num_anchors = len(ratios) * len(scales)

    # 初始化基准anchors (cx,cy,h,w)
    anchors = np.zeros((num_anchors, 4))

    # h,w赋值缩放尺寸
    anchors[:, 2:] = base_size * np.tile(scales, (2, len(ratios))).T

    # 计算anchors的面积
    areas = anchors[:, 2] * anchors[:, 3]

    # 设置长宽比
    anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
    anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

    # 将(x_ctr, y_ctr, w, h) 转为 (x1, y1, x2, y2)
    anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T   # x_ctr-0.5w, w-0.5w
    anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T   # y_ctr-0.5h, h-0.5h

    return anchors


def sliding_anchors_all(shape, stride, anchors):
    """
    根据feature map的大小和步长生成所有的anchors
    :param shape: feature map的形状(H,W)
    :param stride: feature map对应到原图的步长 (SH,SW)
    :param anchors: 基准的anchors(x1,y1,x2,y2)
    :return:
    """
    # 生成anchor的中心点
    shift_x = (np.arange(0, shape[0]) + 0.5) * stride[1]
    shift_y = (np.arange(0, shape[1]) + 0.5) * stride[0]

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel(), shift_y.ravel()
    )).transpose()

    # 使用numpy的广播将 anchor (1, A, 4)
    # 和偏移anchor中心点(K, 1, 4) 相加
    # 最终得到(K, A, 4)
    # 然后再reshape (K*A, 4)
    A = anchors.shape[0]
    K = shifts.shape[0]
    all_anchors = (anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))

    return all_anchors


def pos_neg_iou(pos_overlap, neg_overlap, all_anchors, GT):
    """
    计算all_anchors与给定的GT之间的overlaps，并为每个anchor匹配IOU值最高的gt（IOU值），合理确定阈值来挑选基本的正、负样本
    :param pos_overlap: 
    :param neg_overlap: 
    :param all_anchors: 
    :return: 
    """
    # IOU值计算
    overlaps = overlap(GT.astype(np.float64), all_anchors.astype(np.float64))
    # all_anchors中每个anchor最佳的IOU值与其对应的GT索引
    argmax_iou_index = np.argmax(overlaps.T, axis=1) # （1维数组）按行返回每个anchor对应的IOU值最高的GT索引
    # print(argmax_iou_index)
    argmax_iou_anchor = np.max(overlaps.T, axis=1) # （1维数组）返回每个anchors最好的IOU值
    # 提取正、负样本anchors的索引
    pos_inds = (argmax_iou_anchor >= pos_overlap)
    pos_index = ([i for i, x in enumerate(pos_inds) if x == True]) # IOU值高于0.1的正样本索引
    # print(pos_index)
    neutral_inds = (argmax_iou_anchor > neg_overlap) & ~pos_inds # IOU值介于0.05-0.1之间的中性样本
    neutral_index = ([i for i, x in enumerate(neutral_inds) if x == True])
    # print(neutral_index)
    neg_inds = (argmax_iou_anchor <= neg_overlap)
    neg_index = ([i for i, x in enumerate(neg_inds) if x == True]) # 所有负样本
    # 根据索引提取正、负、中性样本
    pos_sample = np.array([all_anchors[index] for index in pos_index])
    neutral_sample = np.array([all_anchors[index] for index in neutral_index])
    neg_sample = np.array([all_anchors[index] for index in neg_index])
    # print("所有正样本数量：{}".format(len(pos_sample)))
    # print("所有中性样本数量：{}".format(len(neutral_sample)))
    # print("所有负样本数量：{}".format(len(neg_sample)))
    return pos_inds, neutral_inds, argmax_iou_index

def compute_gt_annotations(
    anchors,
    annotations,
    negative_overlap=0.1,
    positive_overlap=0.2
):
    """ 计算anchor和GT的IoU获取正负样本，以及与GT有最高IoU那个anchor

    Args
        anchors: np.array of anchors of shape (N, 4) for (x1, y1, x2, y2).
        annotations: np.array of shape (M, 5) for (x1, y1, x2, y2, label).
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        positive_indices: indices of positive anchors
        ignore_indices: indices of ignored anchors
        argmax_overlaps_inds: ordered overlaps indices
    """

    overlaps = overlap(anchors.astype(np.float64), annotations.astype(np.float64))   # (N,M)维数组，值为IoU
    argmax_overlaps_inds = np.argmax(overlaps, axis=1)    # 每个anchor 最好的IoU的GT的索引号
    max_overlaps = overlaps[np.arange(overlaps.shape[0]), argmax_overlaps_inds]  # 获取每个anchor 最好的IoU值，形状为(N,)

    # 获取正样本和忽略样本anchor索引
    positive_indices = max_overlaps >= positive_overlap  # 正样本索引
    ignore_indices = (max_overlaps > negative_overlap) & ~positive_indices

    return positive_indices, ignore_indices, argmax_overlaps_inds


def bbox_transform(anchors, gt_boxes, mean=None, std=None):
    """
    计算anchor-GT对的边框回归的目标
    :param anchors: 一张图像所有的anchors, (x1,y1,x2,y2)
    :param gt_boxes: anchors对应的gt ,(x1,y1,x2,y2)
    :param mean:
    :param std:
    :return:
    """

    if mean is None:
        mean = np.array([0, 0, 0, 0])
    if std is None:
        std = np.array([0.2, 0.2, 0.2, 0.2])

    if isinstance(mean, (list, tuple)):
        mean = np.array(mean)
    elif not isinstance(mean, np.ndarray):
        raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

    if isinstance(std, (list, tuple)):
        std = np.array(std)
    elif not isinstance(std, np.ndarray):
        raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

    # 计算长宽
    anchor_widths = anchors[:, 2] - anchors[:, 0]
    anchor_heights = anchors[:, 3] - anchors[:, 1]

    # 计算回归目标(左上和右下坐标),没有长宽回归
    targets_dx1 = (gt_boxes[:, 0] - anchors[:, 0]) / anchor_widths   #
    targets_dy1 = (gt_boxes[:, 1] - anchors[:, 1]) / anchor_heights
    targets_dx2 = (gt_boxes[:, 2] - anchors[:, 2]) / anchor_widths  #
    targets_dy2 = (gt_boxes[:, 3] - anchors[:, 3]) / anchor_heights

    targets = np.stack((targets_dx1, targets_dy1, targets_dx2, targets_dy2))
    targets = targets.T

    # 标准化
    targets = (targets - mean) / std

    return targets


def anchor_targets_bbox(
    anchors,
    image_group,
    annotations_group,
    num_classes,
    negative_overlap=0.0,  #
    positive_overlap=0.1   # small
):
    """ 生成一个batch中边框分类和回归的目标

    Args
        anchors: 根据feature map的大小和步长计算好的所有anchors 维度 (N, 4) 的数组 (x1, y1, x2, y2).
        image_group: List of BGR images.
        annotations_group: 标注列表(np.array of shape (N, 5) for (x1, y1, x2, y2, label)).
        num_classes: 类别数 objects + 1.
        negative_overlap: IoU overlap for negative anchors (all anchors with overlap < negative_overlap are negative).
        positive_overlap: IoU overlap or positive anchors (all anchors with overlap > positive_overlap are positive).

    Returns
        labels_batch: batch 包含 labels 和 anchor states (np.array of shape (batch_size, N, num_classes + 1),
                      N是anchors数量， 前num_classes列为类别，最后一列 为anchor state (-1 for ignore, 0 for bg, 1 for fg).
        regression_batch: batch 包含边框回归目标 和 anchor states (np.array of shape (batch_size, N, 4 + 1),
                      N是anchors数量，前4列为(x1, y1, x2, y2) 的回归目标，最后一列为 anchor states (-1 for ignore, 0 for bg, 1 for fg).
        boxes_batch: anchor对应的GT边框信息 (np.array of shape (batch_size, N, 5), where N is the number of anchors for an image)
    """

    assert (len(image_group) == len(annotations_group)), "The length of the images and annotations need to be equal."
    assert (len(annotations_group) > 0), "No data received to compute anchor targets for."

    batch_size = len(image_group)
    # 最后一位为标志位 -1 ignore, 0 negtive, 1 postive
    regression_batch = np.zeros((batch_size, anchors.shape[0], 4 + 1), dtype=keras.backend.floatx())
    labels_batch = np.zeros((batch_size, anchors.shape[0], num_classes + 1), dtype=keras.backend.floatx())
    boxes_batch = np.zeros((batch_size, anchors.shape[0], annotations_group[0].shape[1]), dtype=keras.backend.floatx())

    # compute labels and regression targets
    for index, (image, annotations) in enumerate(zip(image_group, annotations_group)):
        if annotations.shape[0]:
            # obtain indices of gt annotations with the greatest overlap
            positive_indices, ignore_indices, argmax_overlaps_inds = compute_gt_annotations(anchors, annotations, negative_overlap, positive_overlap)
            # 全部初始化为忽略
            labels_batch[index, :, -1] = -1
            labels_batch[index, positive_indices, -1] = 1

            regression_batch[index, :, -1] = -1
            regression_batch[index, positive_indices, -1] = 1

            # compute box regression targets
            annotations = annotations[argmax_overlaps_inds]  # 每个anchor对应的GT
            boxes_batch[index, ...] = annotations  # 记录类别

            # 计算目标类别，默认都是背景类
            labels_batch[index, positive_indices, annotations[positive_indices, 4].astype(int)] = 1  # 赋值标记位
            # 计算回归目标
            regression_batch[index, :, :-1] = bbox_transform(anchors, annotations)

        # 按照1:3 正负样本比启发式采样
        postive_num = np.sum(labels_batch[index, :, -1] == 1)
        for i in np.random.randint(0, anchors.shape[0], 2 * postive_num):
            if not (labels_batch[index, :, -1]-1).all():
                labels_batch[index, i, -1] = 0   # 设为背景类
                regression_batch[index, i, -1] = 0

        # 忽略的
        labels_batch[index, ignore_indices, -1] = -1
        regression_batch[index, ignore_indices, -1] = -1

    # 返回_batch数组中，对应样本的下标索引
    inds = (labels_batch[:, :, -1] != -1) # 所有正负样本（排除背景样本）
    pos_inds = (labels_batch[:, :, -1] == 1) # 所有正样本（排除非正样本）
    # 打印正负样本数量
    if np.random.rand() < 0.002:
        print("post_num:{},bg_num:{},ignore_num:{}".format(np.sum(labels_batch[:, :, -1] == 1),
                                                           np.sum(labels_batch[:, :, -1] == 0),
                                                           np.sum(labels_batch[:, :, -1] == -1)))
    '''
    # batch打平
    inds = inds.ravel()
    labels_batch = labels_batch.reshape(batch_size*anchors.shape[0], num_classes + 1)[np.newaxis, :, :]
    regression_batch = regression_batch.reshape(batch_size*anchors.shape[0], 4 + 1)[np.newaxis, :, :]
    '''
    return labels_batch, regression_batch, boxes_batch, inds, pos_inds
