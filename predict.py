import cv2
import numpy as np
import pickle
import time
from keras import backend as K
from keras.layers import Input
from keras.models import Model
import roi_helpers
import argparse
import os
import resnet as nn
from visualize import draw_boxes_and_label_on_image


def format_img_size(img, cfg):
    """ 缩放图片尺寸，短边为600 """
    img_min_side = float(cfg.im_size)
    (height, width, _) = img.shape

    if width <= height:
        ratio = img_min_side / width
        new_height = int(ratio * height)
        new_width = int(img_min_side)
    else:
        ratio = img_min_side / height
        new_width = int(ratio * width)
        new_height = int(img_min_side)
    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    return img, ratio


def format_img_channels(img, cfg):
    """ 每个channel减去像素均值，将channel放在第一维，然后在前面增加一个维度 """
    img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)
    img[:, :, 0] -= cfg.img_channel_mean[0]
    img[:, :, 1] -= cfg.img_channel_mean[1]
    img[:, :, 2] -= cfg.img_channel_mean[2]
    img /= cfg.img_scaling_factor
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    img, ratio = format_img_size(img, C)
    img = format_img_channels(img, C)
    return img, ratio


def get_real_coordinates(ratio, x1, y1, x2, y2):
    """
    将坐标值从resize后的图片映射到原始图片
    """
    real_x1 = int(round(x1 // ratio))
    real_y1 = int(round(y1 // ratio))
    real_x2 = int(round(x2 // ratio))
    real_y2 = int(round(y2 // ratio))

    return real_x1, real_y1, real_x2, real_y2


def predict_single_image(img_path, model_rpn, model_classifier, cfg, class_mapping):
    """
    预测单张图片
    :param img_path: 图片路径
    :param model_rpn: rpn模型
    :param model_classifier: 目标检测模型
    :param cfg: 配置参数
    :param class_mapping: 类别映射
    :return:
    """
    st = time.time()
    img = cv2.imread(img_path)  # 读取图片
    if img is None:
        print('reading image failed.')
        exit(0)

    X, ratio = format_img(img, cfg)  # 预处理图片（缩放、变换维度）
    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))

    # 得到所有anchor的分类得分、回归参数以及feature map
    [Y1, Y2, F] = model_rpn.predict(X)

    # 得到proposals
    result = roi_helpers.rpn_to_roi(Y1, Y2, cfg, K.image_dim_ordering(), overlap_thresh=0.7)

    # convert from (x1,y1,x2,y2) to (x1,y1,w,h)
    result[:, 2] -= result[:, 0]
    result[:, 3] -= result[:, 1]

    bbox_threshold = 0.8
    # 存放每个类别的所有边框
    boxes = {}
    # 按批处理rois，batch siez为32，“+1”是为了处理最后一个批次
    for jk in range(result.shape[0] // cfg.num_rois + 1):
        # 增加维度，shape(1,32,4)
        rois = np.expand_dims(result[cfg.num_rois * jk:cfg.num_rois * (jk + 1), :], axis=0)
        if rois.shape[1] == 0:
            break
        if jk == result.shape[0] // cfg.num_rois:
            # 由于最后一个batch中roi的个数可能小于32，这里将填满到32个，
            # 并将填充的部分用当前batch的第一个roi信息填充
            curr_shape = rois.shape
            target_shape = (curr_shape[0], cfg.num_rois, curr_shape[2])
            rois_padded = np.zeros(target_shape).astype(rois.dtype)
            rois_padded[:, :curr_shape[1], :] = rois
            rois_padded[0, curr_shape[1]:, :] = rois[0, 0, :]
            rois = rois_padded

        # 得到当前batch类别预测得分和相应的回归参数
        # p_cls(1,32,cls_nums+bg), p_regr(1,32,cls_nums*4)
        p_cls, p_regr = model_classifier.predict([F, rois])

        # 遍历每个roi
        for ii in range(p_cls.shape[1]):
            # 如果当前roi为某类的最大概率小于阈值，或者最大概率对应的是背景类，则丢弃
            if np.max(p_cls[0, ii, :]) < bbox_threshold or np.argmax(p_cls[0, ii, :]) == (p_cls.shape[2] - 1):
                continue
            cls_num = np.argmax(p_cls[0, ii, :])  # 最大概率对应的类别数字
            if cls_num not in boxes.keys():
                boxes[cls_num] = []

            x, y, w, h = rois[0, ii, :]  # 当前roi的左上角坐标和宽高
            try:
                # 得到回归参数
                (tx, ty, tw, th) = p_regr[0, ii, 4 * cls_num:4 * (cls_num + 1)]
                # 下面四行对应了训练过程中乘以cfg.classifier_regr_std
                tx /= cfg.classifier_regr_std[0]
                ty /= cfg.classifier_regr_std[1]
                tw /= cfg.classifier_regr_std[2]
                th /= cfg.classifier_regr_std[3]
                # 调整边框位置
                x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
            except Exception as e:
                print(e)
                pass

            # 将roi从feature map映射回resize后的图片，存入boxes
            boxes[cls_num].append(
                [cfg.rpn_stride * x, cfg.rpn_stride * y, cfg.rpn_stride * (x + w), cfg.rpn_stride * (y + h),
                 np.max(p_cls[0, ii, :])])

    for cls_num, box in boxes.items():
        # nms
        boxes_nms = roi_helpers.non_max_suppression_fast(box, overlap_thresh=0.3)
        boxes[cls_num] = boxes_nms
        print("【{}】:".format(class_mapping[cls_num]))
        for b in boxes_nms:
            # 将坐标映射回原始图片
            b[0], b[1], b[2], b[3] = get_real_coordinates(ratio, b[0], b[1], b[2], b[3])
            print('coordinate:{} prob: {}'.format(b[0: 4], b[-1]))

    # 绘图
    img = draw_boxes_and_label_on_image(img, class_mapping, boxes)
    print('Elapsed time: {}'.format(time.time() - st))
    cv2.imshow('image', img)
    result_path = './result_images/{}.jpg'.format(os.path.basename(img_path).split('.')[0])
    print('result saved into ', result_path)
    cv2.imwrite(result_path, img)
    cv2.waitKey(0)


def predict(args_):
    """
    预测图片
    :param args_: 从命令行获取的参数
    :return:
    """
    path = args_.path  # 图片路径
    # 加载配置文件
    with open('config.pickle', 'rb') as f_in:
        cfg = pickle.load(f_in)
    # cfg.use_horizontal_flips = False
    # cfg.use_vertical_flips = False
    # cfg.rot_90 = False

    class_mapping = cfg.class_mapping
    if 'bg' not in class_mapping:
        class_mapping['bg'] = len(class_mapping)
    class_mapping = {v: k for k, v in class_mapping.items()}  # 键值互换

    input_shape_img = (None, None, 3)
    input_shape_features = (None, None, 1024)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(cfg.num_rois, 4))
    feature_map_input = Input(shape=input_shape_features)

    # 定义基础网络
    shared_layers = nn.nn_base(img_input, trainable=True)

    # 定义RPN
    num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    # 定义检查网络
    classifier = nn.classifier(feature_map_input, roi_input, cfg.num_rois, nb_classes=len(class_mapping),
                               trainable=True)

    model_rpn = Model(img_input, rpn_layers)
    # model_classifier_only = Model([feature_map_input, roi_input], classifier)
    model_classifier = Model([feature_map_input, roi_input], classifier)

    # 加载权重
    print('Loading weights from {}'.format(cfg.model_path))
    model_rpn.load_weights(cfg.model_path, by_name=True)
    model_classifier.load_weights(cfg.model_path, by_name=True)
    # 编译模型
    model_rpn.compile(optimizer='sgd', loss='mse')
    model_classifier.compile(optimizer='sgd', loss='mse')

    if os.path.isdir(path):
        for idx, img_name in enumerate(sorted(os.listdir(path))):
            if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
                continue
            print(img_name)
            predict_single_image(os.path.join(path, img_name), model_rpn,
                                 model_classifier, cfg, class_mapping)
    elif os.path.isfile(path):
        print('predict image from {}'.format(path))
        predict_single_image(path, model_rpn, model_classifier, cfg, class_mapping)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', default='VOC2007_test/JPEGImages/000010.jpg', help='image path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    predict(args)
