# -*- coding: utf-8 -*-
"""
Created on 2018/11/6 17:20

@author: royce.mao

# 1st-stage的RPN网络生成的既定个数的proposals，在原图上做crops并做长、宽既定的resize，提取2nd-stage网络的图片数据输入（考虑了batch维度）
"""
import tensorflow as tf
import keras.backend as K
from overlap_2nd import overlap
from PIL import Image
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np

def props_pic(sess, proposals, all_tag_annos, all_num_annos, all_imgs):
    """
    RPN网络生成的proposals映射回原图，并修正crop区域含有的小数字坐标，进行第2阶段的输入训练。
    :param proposals: （batch_size，num_rois，4）
    :param num_rois: 
    :param all_imgs: 
    :return: rs_pic（4维数组）, rs_wh（2维数组）
    """
    rs_pic = []
    rs_boxes = []
    rs_num_gt_pic = []
    rs_wh = []
    gt_index = []
    all_tag_annos = np.array(all_tag_annos[0][:4], dtype=np.float64)[np.newaxis, :]
    for i, img in enumerate(all_imgs):
        rs_pic_single = [] # 单独一张图片的img crop numpy
        rs_num_gt_single = [] # 单独一张图片crop得到的区域，所对应的小数字标注
        gt_index_single = []
        # 统一resize到（80，40，3）
        width_mean = 80
        height_mean = 160
        # 保留得分前5的proposals
        rs_boxes_single = proposals[i]
        # 把所有保存的proposals做padding（这里直接做长宽扩充）
        for j, prop in enumerate(proposals[i]):
            prop = prop[np.newaxis, :]
            num_labels = np.zeros((len(all_num_annos[i]), 5))
            '''
            if overlap(all_tag_annos[i][:4][np.newaxis, :], prop)[0][0] != 0: # np.min(prop[1], all_tag_annos[1])
                y1 = np.min((prop[0][1], all_tag_annos[i][1]))
                y2 = np.max((prop[0][3], all_tag_annos[i][3]))
                x1 = np.min((prop[0][0], all_tag_annos[i][0]))
                x2 = np.max((prop[0][2], all_tag_annos[i][2]))
            '''
            # 短边为600的resize图片上，1阶段的Top5 proposals 高宽均值大约为（40，20），padding宽加10，高加20个像素点
            rs_boxes_single[j] = [prop[0][0]-7,prop[0][1]-15,prop[0][2]+7,prop[0][3]+15]
            # print(overlap(all_tag_annos[i][:4][np.newaxis, :], rs_boxes_single[j][np.newaxis, :])[0][0])
            # 然后进行小数字坐标标签的关系变换
            ## crops后的图片num_labels坐标等于原坐标减去crop_prop的左上角坐标值(x_min,y_min)
            x_min = rs_boxes_single[j][0]
            y_min = rs_boxes_single[j][1]
            # 判断padding后的proposals是否包含有Tags标签
            if all_tag_annos[i][:4][0]>rs_boxes_single[j][0] and all_tag_annos[i][:4][1]>rs_boxes_single[j][1] and all_tag_annos[i][:4][2]<rs_boxes_single[j][2] and all_tag_annos[i][:4][3]<rs_boxes_single[j][3]:
                gt_index_single.append(j)
                ## proposal对应（960， 540）图片经crops的号码牌数字gt坐标
                num_labels[:, 0] = np.array((all_num_annos[i][:, 0]), dtype=np.float64) - x_min
                num_labels[:, 2] = np.array((all_num_annos[i][:, 2]), dtype=np.float64) - x_min
                num_labels[:, 1] = np.array((all_num_annos[i][:, 1]), dtype=np.float64) - y_min
                num_labels[:, 3] = np.array((all_num_annos[i][:, 3]), dtype=np.float64) - y_min
                num_labels[:, 4] = all_num_annos[i][:, 4]
                # 每张图crops后做统一resize的gt坐标变化
            num_labels[:, 0] = num_labels[:, 0] * (width_mean/(rs_boxes_single[j][2]-rs_boxes_single[j][0]))
            num_labels[:, 2] = num_labels[:, 2] * (width_mean/(rs_boxes_single[j][2]-rs_boxes_single[j][0]))
            num_labels[:, 1] = num_labels[:, 1] * (height_mean/(rs_boxes_single[j][3]-rs_boxes_single[j][1]))
            num_labels[:, 3] = num_labels[:, 3] * (height_mean/(rs_boxes_single[j][3]-rs_boxes_single[j][1]))
            rs_num_gt_single.append(num_labels)
        # 然后再统一进行对应区域图片数据的crops
        for j, prop in enumerate(rs_boxes_single): # prop[1],prop[0],prop[3],prop[2]
            # 这里涉及到张量的计算过程，而且位于循环体中，导致迭代过程中内存暴增
            '''
            a = tf.image.crop_and_resize(img, [[prop[1]/600,prop[0]/1066,prop[3]/600,prop[2]/1066]], box_ind=[0], crop_size=[height_mean, width_mean])
            b = a.eval(session=sess)
            '''
            # 所以，重新定义img的crops和resize的代码
            image = Image.fromarray(img[0].astype('uint8')) # 必须加上.astype('uint8')
            prop_crop = image.crop([prop[0], prop[1], prop[2], prop[3]])
            prop_crop = img_to_array(prop_crop)
            prop_pic = cv2.resize(prop_crop, (width_mean, height_mean)) # 图片标准化resize
            rs_pic_single.append(prop_pic)
                # rs_pic_single.append(img.crop([img.size[0] / 4, img.size[1] / 4, img.size[0] * 3 / 4, img.size[1] * 3 / 4]))
        rs_pic.append(rs_pic_single)
        rs_boxes.append(rs_boxes_single)
        rs_wh.append([height_mean, width_mean])
        rs_num_gt_pic.append(rs_num_gt_single)
        gt_index.append(gt_index_single)
    return rs_pic, rs_boxes, rs_num_gt_pic, rs_wh, gt_index
