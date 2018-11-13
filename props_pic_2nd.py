# -*- coding: utf-8 -*-
"""
Created on 2018/11/6 17:20

@author: royce.mao

# 1st-stage的RPN网络生成的既定个数的proposals，在原图上做crops并做长、宽既定的resize，提取2nd-stage网络的图片数据输入（考虑了batch维度）
"""
import tensorflow as tf
import keras.backend as K
from overlap_2nd import overlap
import numpy as np

def props_pic(sess, proposals, all_tag_annos, all_imgs):
    """
    RPN网络生成的proposals映射回原图，进行第2阶段的输入crops。
    :param proposals: （batch_size，num_rois，4）
    :param num_rois: 
    :param all_imgs: 
    :return: rs_pic（4维数组）, rs_wh（2维数组）
    """
    rs_pic = []
    rs_boxes = []
    rs_wh = []
    all_tag_annos = np.array(all_tag_annos[0][:4], dtype=np.float64)[np.newaxis, :]
    for i, img in enumerate(all_imgs):
        rs_pic_single = []
        rs_boxes_single = []
        max = 0
        # 统一resize到（80，40，3）
        width_mean = 20
        height_mean = 40
        for prop in proposals[i]:
            prop = prop[np.newaxis, :]
            # 首先寻找与tag框有交集的proposals，然后取两者交集区域的上下限作为crops的图像
            # print(overlap(all_tag_annos[i][:4][np.newaxis, :], prop))
            if overlap(all_tag_annos[i][:4][np.newaxis, :], prop)[0][0] > max: # np.min(prop[1], all_tag_annos[1])
                max = overlap(all_tag_annos[i][:4][np.newaxis, :], prop)[0][0]
                y1 = np.min((prop[0][1], all_tag_annos[i][1]))
                y2 = np.max((prop[0][3], all_tag_annos[i][3]))
                x1 = np.min((prop[0][0], all_tag_annos[i][0]))
                x2 = np.max((prop[0][2], all_tag_annos[i][2]))
                a = tf.image.crop_and_resize(img, [[y1,x1,y2,x2]], box_ind=[0], crop_size=[height_mean, width_mean])
                b = a.eval(session=sess)
                rs_pic_single = b
                # rs_pic_single.append(img.crop([img.size[0] / 4, img.size[1] / 4, img.size[0] * 3 / 4, img.size[1] * 3 / 4]))
                rs_boxes_single = [x1, y1, x2, y2]
        rs_pic.append(rs_pic_single)
        rs_boxes.append(rs_boxes_single)
        rs_wh.append([height_mean, width_mean])
    return rs_pic, rs_boxes, rs_wh

def pic_num_label(rs_pic, rs_boxes, rs_wh, all_num_annos):
    """
    第2阶段crops后的号码牌数字标签坐标与原数字标签坐标之间的关系
    :param proposals: 
    :param rs_pic: 
    :param rs_wh: 
    :param all_num_annos: 
    :return: 4维数组（batch_size，num_props，3，5）
    """
    num_labels_all = []
    for i, img_props in enumerate(rs_boxes):
        # 初始化一张图片的数字标签集合list
        num_labels_pic = []
        # 首先判断下某张图片的proposals是否都越界
        if len(rs_boxes[i]) != 0:
            # 初始化一张图片中单个小proposal的数字标签
            num_labels = np.zeros((len(all_num_annos[i]), 5))
            # crops后的图片num_labels坐标等于原坐标减去crop_prop的左上角坐标值
            x_min = img_props[0]
            y_min = img_props[1]
            # proposal对应（960， 540）图片经crops的号码牌数字gt坐标
            num_labels[:, 0] = np.array((all_num_annos[i][:, 0]), dtype=np.float64) - x_min
            num_labels[:, 2] = np.array((all_num_annos[i][:, 2]), dtype=np.float64) - x_min
            num_labels[:, 1] = np.array((all_num_annos[i][:, 1]), dtype=np.float64) - y_min
            num_labels[:, 3] = np.array((all_num_annos[i][:, 3]), dtype=np.float64) - y_min
            num_labels[:, 4] = all_num_annos[i][:, 4]
            # 每张图crops后做统一resize的gt坐标变化
            num_labels[:, 0] = num_labels[:, 0] * (rs_wh[i][1]/(img_props[2]-img_props[0]))
            num_labels[:, 2] = num_labels[:, 2] * (rs_wh[i][1]/(img_props[2]-img_props[0]))
            num_labels[:, 1] = num_labels[:, 1] * (rs_wh[i][0]/(img_props[3]-img_props[1]))
            num_labels[:, 3] = num_labels[:, 3] * (rs_wh[i][0]/(img_props[3]-img_props[1]))
            num_labels_pic.append(num_labels)
        num_labels_all.append(np.array(num_labels_pic))
    return num_labels_all
