import cv2
import numpy as np
import pickle
from keras import backend as K
from keras.layers import Input
from keras.models import Model
import roi_helpers
import argparse
import os
import resnet as nn
from visualize import draw_boxes_and_label_on_image
from net_design_2nd import stage_2_net_vgg
from anchor_2nd import anchors_generation, sliding_anchors_all
from PIL import Image
from keras.preprocessing.image import img_to_array


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
    # img = img[:, :, (2, 1, 0)]
    img = img.astype(np.float32)

    img[:, :, 0] -= np.mean(img[:, :, 0])
    img[:, :, 1] -= np.mean(img[:, :, 1])
    img[:, :, 2] -= np.mean(img[:, :, 2])

    img /= cfg.img_scaling_factor
    # img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


def format_img(img, C):
    """ formats an image for model prediction based on config """
    ratio = 1
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
    img = cv2.imread(img_path)  # 读取图片

    if img is None:
        print('reading image failed.')
        exit(0)

    # print(class_mapping)

    X, ratio = format_img(img, cfg)  # 预处理图片（缩放、变换维度）
    '''
    if K.image_dim_ordering() == 'tf':
        X = np.transpose(X, (0, 2, 3, 1))
    '''
    # 得到所有anchor的分类得分、回归参数以及feature map
    P_rpn = model_rpn.predict_on_batch(X)

    # 得到proposals (rois)
    proposals = roi_helpers.rpn_to_roi(P_rpn[0], P_rpn[1], cfg, K.image_dim_ordering(), use_regr=True,
                                                overlap_thresh=0.7,
                                                max_boxes=5)

    rpn_outputs = []  # 存放proposals

    score = 5
    # 将proposal坐标从feature map映射回输入图片
    for proposal in proposals:
        rpn_outputs.append(
            [cfg.rpn_stride * proposal[0], cfg.rpn_stride * proposal[1],
             cfg.rpn_stride * proposal[2], cfg.rpn_stride * proposal[3], score])
        score -= 1

    # for box in rpn_outputs:
    # nms
    boxes_nms = roi_helpers.non_max_suppression_fast(rpn_outputs, overlap_thresh=0.8)
    rpn_outputs = boxes_nms
    print("【RPN outputs】:")
    for b in boxes_nms:
        # 将坐标映射回原始图片
        b[0], b[1], b[2], b[3] = get_real_coordinates(ratio, b[0], b[1], b[2], b[3])
        print('coordinate:{} prob: {}'.format(b[0: 4], b[-1]))

    # 将rois从原图中裁剪出来，并记录裁剪尺度和裁剪比例
    image = Image.fromarray(img.astype('uint8'))
    resized_width = 80
    resized_height = 160
    imgs_crop = []  # 存放裁剪后的图片
    crop_scales = []  # 存放每个roi从原图中的裁剪尺度
    crop_ratios = []  # 存放每个roi的缩放比
    for roi in rpn_outputs:
        w = roi[2] - roi[0]
        h = roi[3] - roi[1]
        ratio_w = resized_width / w
        ratio_h = resized_height / h
        crop_ratios.append([ratio_w, ratio_h])
        crop_scales.append([round(roi[0]), round(roi[1])])
        prop_crop = image.crop([roi[0], roi[1], roi[2], roi[3]])
        prop_crop = img_to_array(prop_crop)
        prop_pic = cv2.resize(prop_crop, (resized_width, resized_height))
        imgs_crop.append(prop_pic)

    base_anchors = anchors_generation(16, [0.5 ** (1.0 / 3.0), 1, 2 ** (1.0 / 3.0)],
                                      [0.5, 0.5 ** (1.0 / 2.0), 1, 2 ** (1.0 / 3.0), 2 ** (1.0 / 2.0), 2])
    all_anchors = sliding_anchors_all((10, 20), (8, 8), base_anchors)

    final_boxes = {}
    for i, img_crop in enumerate(imgs_crop):
        p_cls, p_regr = model_classifier.predict_on_batch(img_crop[np.newaxis, :, :, :])
        boxes = {}
        bbox_threshold = 0.5
        # 遍历每个anchor
        for ii in range(p_cls.shape[1]):
            # 如果当前anchor为某类的最大概率小于阈值，或者最大概率对应的是背景类，则丢弃
            if np.max(p_cls[0, ii, :]) < bbox_threshold or np.argmax(p_cls[0, ii, :]) == (p_cls.shape[2] - 1):
                continue
            cls_num = np.argmax(p_cls[0, ii, :])  # 最大概率类别对应的下标
            if cls_num not in boxes.keys():
                boxes[cls_num] = []

            # 边框回归
            x1, y1, x2, y2 = regr_revise(all_anchors[ii, :], p_regr[0, ii, 4*cls_num: 4*(cls_num+1)])
            boxes[cls_num].append([all_anchors[ii, 0], all_anchors[ii, 1],
                                   all_anchors[ii, 2], all_anchors[ii, 3], np.max(p_cls[0, ii, :])])
            # boxes[cls_num].append([x1, y1, x2, y2, np.max(p_cls[0, ii, :])])

            # print('================>')
            # print('回归前坐标：{}'.format(all_anchors[ii, :]))
            # print('回归参数：{}'.format(p_regr[0, ii, 4*cls_num: 4*(cls_num+1)]))
            # print('回归后坐标：{}'.format([x1, y1, x2, y2]))
            # print('<================')

        for cls_num, box in boxes.items():
            # nms
            boxes_nms = roi_helpers.non_max_suppression_fast(box, overlap_thresh=0.3, max_boxes=1)
            boxes[cls_num] = boxes_nms
            for b in boxes_nms:
                # 将坐标映射回原始小图片
                b[0] = round(b[0] / crop_ratios[i][0])
                b[1] = round(b[1] / crop_ratios[i][1])
                b[2] = round(b[2] / crop_ratios[i][0])
                b[3] = round(b[3] / crop_ratios[i][1])
                # 将坐标映射回原始大图片
                b[0] += crop_scales[i][0]
                b[1] += crop_scales[i][1]
                b[2] += crop_scales[i][0]
                b[3] += crop_scales[i][1]

                print('【{}】'.format(class_mapping[cls_num]))
                print('coordinate:{} prob: {}'.format(b[0: 4], b[-1]))
                if cls_num not in final_boxes.keys():
                    final_boxes[cls_num] = []
                final_boxes[cls_num].append([b[0], b[1], b[2], b[3], b[4]])

    # 绘图保存
    img = draw_boxes_and_label_on_image(img, class_mapping, final_boxes)
    result_path = './result_images/{}.png'.format(os.path.basename(img_path).split('.')[0])
    print('result saved into ', result_path)
    cv2.imwrite(result_path, img)


def regr_revise(anchor, regr):
    """
    第1阶段bbox_transform函数定义的回归目标在4个偏移量(dx,dy,dw,dh)基础上，做位置修正
    :return:
    """
    x_target_center = regr[0] * (anchor[2] - anchor[0]) + (anchor[2] + anchor[0]) / 2.0
    y_target_center = regr[1] * (anchor[3] - anchor[1]) + (anchor[3] + anchor[1]) / 2.0
    w_target = np.exp(regr[2]) * (anchor[2] - anchor[0])
    h_target = np.exp(regr[3]) * (anchor[3] - anchor[1])
    x1_target = x_target_center - w_target / 2.0
    y1_target = y_target_center - h_target / 2.0
    x2_target = x_target_center + w_target / 2.0
    y2_target = y_target_center + h_target / 2.0
    return x1_target, y1_target, x2_target, y2_target


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
    img_input = Input(shape=input_shape_img)

    # 定义基础网络
    shared_layers = nn.nn_base(img_input, trainable=True)

    # 定义RPN
    num_anchors = len(cfg.anchor_box_scales) * len(cfg.anchor_box_ratios)
    rpn_layers = nn.rpn(shared_layers, num_anchors)

    # 定义检查网络
    small_img_input = Input(shape=(160, 80, 3))
    classifier = stage_2_net_vgg(len(class_mapping), small_img_input)

    model_rpn = Model(img_input, rpn_layers)
    model_classifier = Model(small_img_input, classifier)

    # 加载权重
    model_rpn.load_weights('model_trained/model_final.hdf5', by_name=True)
    model_classifier.load_weights('model_trained/model_final.hdf5', by_name=True)

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
    # 00020_annotated_num/images/aug_3_012.png
    parser.add_argument('--path', '-p', default='./00020_annotated_num_120/images', help='image path')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    predict(args)
