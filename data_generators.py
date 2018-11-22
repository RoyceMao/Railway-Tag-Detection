import numpy as np
import random
import cv2
import data_augment


def union(a, b, area_inter):
    """
    计算box a与box b的并集
    """
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_union = area_a + area_b - area_inter
    return area_union


def intersection(a, b):
    """
    计算box a与box b的交集
    """
    x = max(a[0], b[0])
    y = max(a[1], b[1])
    w = min(a[2], b[2]) - x
    h = min(a[3], b[3]) - y
    if w < 0 or h < 0:
        return 0
    return w * h


def iou(a, b):
    """
    计算box a与box b的交并比, [x1,y1,x2,y2]
    """
    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0

    area_inter = intersection(a, b)  # 交集面积
    area_union = union(a, b, area_inter)  # 并集面积

    return area_inter / area_union


def get_new_img_size(width, height, img_min_side=600):
    """
    对原始图片做resize
    :param width: int, 原始图片宽
    :param height: int, 原始图片高
    :param img_min_side: int, 短边大小
    :return: resize后的宽和高
    """
    if width <= height:
        f = float(img_min_side) / width
        resized_height = int(f * height)
        resized_width = img_min_side
    else:
        f = float(img_min_side) / height
        resized_width = int(f * width)
        resized_height = img_min_side

    return resized_width, resized_height


def calc_rpn(C, img_data, width, height, resized_width, resized_height, img_length_calc_function):
    """
    生成anchors，并标定正负样本以及计算回归系数
    :param C: 配置参数
    :param img_data: dict, 包含了原始图片信息
    :param width: 原始图片宽
    :param height: 原始图片长
    :param resized_width: resize后的宽
    :param resized_height: resize后的长
    :param img_length_calc_function: 将图片尺寸转换到feature map尺寸的一个函数
    :return: 标定好了的anchor以及回归系数
    """
    downscale = float(C.rpn_stride)  # 输入图片到feature map的下采样尺度，16
    anchor_sizes = C.anchor_box_scales  # anchor尺寸，[128, 256, 512]
    anchor_ratios = C.anchor_box_ratios  # anchor宽高比，[[1,1], [1,2], [2,1]]
    num_anchors = len(anchor_sizes) * len(anchor_ratios)  # 基准anchor数量，9

    # 得到feature map的size
    output_width, output_height = img_length_calc_function(resized_width, resized_height)

    n_anchratios = len(anchor_ratios)  # 3

    # 初始化
    y_rpn_overlap = np.zeros((output_height, output_width, num_anchors))  # shape(m,n,9), 记录anchor是正样本还是负样本
    y_is_box_valid = np.zeros((output_height, output_width, num_anchors))  # shape(m,n,9), 记录anchor是否有效（在训练中是否起作用)
    y_rpn_regr = np.zeros((output_height, output_width, num_anchors * 4))  # shape(m,n,36), 记录anchor的回归系数

    # num_bboxes = len(img_data['bboxes'])  # 该图片中gt-box的数量
    num_bboxes = len(img_data['outer_boxes'])

    num_anchors_for_bbox = np.zeros(num_bboxes).astype(int)  # shape(num_bboxes,), 记录每个gt-box包含正anchor的个数
    best_anchor_for_bbox = -1 * np.ones((num_bboxes, 4)).astype(int)  # shape(num_bboxes,4), 记录每个gt-box的最佳anchor
    best_iou_for_bbox = np.zeros(num_bboxes).astype(np.float32)  # shape(num_bboxes,), 记录每个gt-box的最佳iou
    best_x_for_bbox = np.zeros((num_bboxes, 4)).astype(int)  # shape(num_bboxes,4), 记录每个gt-box的最佳anchor坐标
    best_dx_for_bbox = np.zeros((num_bboxes, 4)).astype(np.float32)  # shape(num_bboxes,4), 记录每个gt-box的最佳anchor的偏移系数

    # 由于原图进行了resize缩放，所有原图中的gt-box也要进行缩放
    gta = np.zeros((num_bboxes, 4))
    for bbox_num, bbox in enumerate(img_data['outer_boxes']):
        gta[bbox_num, 0] = bbox['x1'] * (resized_width / float(width))
        gta[bbox_num, 1] = bbox['x2'] * (resized_width / float(width))
        gta[bbox_num, 2] = bbox['y1'] * (resized_height / float(height))
        gta[bbox_num, 3] = bbox['y2'] * (resized_height / float(height))

    # 遍历所有的anchor size和anchor ratio
    for anchor_size_idx in range(len(anchor_sizes)):
        for anchor_ratio_idx in range(n_anchratios):
            anchor_x = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][0]  # 当前anchor的宽
            anchor_y = anchor_sizes[anchor_size_idx] * anchor_ratios[anchor_ratio_idx][1]  # 当前anchor的高

            # 将feature map上的每个点都作为锚点，并映射到输入图片，再计算出anchor坐标
            for ix in range(output_width):
                # 当前anchor在输入图片中的x坐标
                x1_anc = downscale * (ix + 0.5) - anchor_x / 2
                x2_anc = downscale * (ix + 0.5) + anchor_x / 2

                # 忽略超出图片尺寸的anchor
                if x1_anc < 0 or x2_anc > resized_width:
                    continue

                for jy in range(output_height):

                    # 当前anchor在输入图片中的y坐标
                    y1_anc = downscale * (jy + 0.5) - anchor_y / 2
                    y2_anc = downscale * (jy + 0.5) + anchor_y / 2

                    # 忽略超出图片尺寸的anchor
                    if y1_anc < 0 or y2_anc > resized_height:
                        continue

                    # anchor_type 说明该anchor是否为正样本，默认为负样本
                    anchor_type = 'neg'

                    # 记录当前anchor与所有gt-box中的最佳iou
                    best_iou_for_loc = 0.0

                    # 遍历所有的gt-box
                    for bbox_num in range(num_bboxes):
                        # 计算当前anchor与当前gt-box的iou
                        curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
                                       [x1_anc, y1_anc, x2_anc, y2_anc])
                        # 如果iou大于当前gt-box的最佳iou值或者大于预设的阈值，则计算回归系数
                        if curr_iou > best_iou_for_bbox[bbox_num] or curr_iou > C.rpn_max_overlap:
                            cx = (gta[bbox_num, 0] + gta[bbox_num, 1]) / 2.0
                            cy = (gta[bbox_num, 2] + gta[bbox_num, 3]) / 2.0
                            cxa = (x1_anc + x2_anc) / 2.0
                            cya = (y1_anc + y2_anc) / 2.0

                            tx = (cx - cxa) / (x2_anc - x1_anc)
                            ty = (cy - cya) / (y2_anc - y1_anc)
                            tw = np.log((gta[bbox_num, 1] - gta[bbox_num, 0]) / (x2_anc - x1_anc))
                            th = np.log((gta[bbox_num, 3] - gta[bbox_num, 2]) / (y2_anc - y1_anc))

                        if img_data['outer_boxes'][bbox_num]['class'] != 'bg':
                            # 每个gt-box都要有个与其对应的anchor，需要跟踪记录哪个anchor是最好的（根据iou）
                            if curr_iou > best_iou_for_bbox[bbox_num]:  # 如果大于当前gt-box的最佳iou，则做下面4个更新
                                best_anchor_for_bbox[bbox_num] = [jy, ix, anchor_ratio_idx, anchor_size_idx]
                                best_iou_for_bbox[bbox_num] = curr_iou
                                best_x_for_bbox[bbox_num, :] = [x1_anc, x2_anc, y1_anc, y2_anc]
                                best_dx_for_bbox[bbox_num, :] = [tx, ty, tw, th]

                            # 如果iou大于预设的阈值，则将当前的anchor设为正样本
                            if curr_iou > C.rpn_max_overlap:
                                anchor_type = 'pos'
                                num_anchors_for_bbox[bbox_num] += 1  # 当前gt-box的正anchor数量加一
                                # 如果iou是当前anchor的最佳值，则更新当前anchor的最佳iou以及回归系数
                                if curr_iou > best_iou_for_loc:
                                    best_iou_for_loc = curr_iou
                                    best_regr = (tx, ty, tw, th)

                            # 如果iou在最大和最小的阈值之间，且当前anchor不为正样本，则将其设为中性样本
                            if (C.rpn_min_overlap < curr_iou < C.rpn_max_overlap) and anchor_type != 'pos':
                                    anchor_type = 'neutral'

                    # 根据anchor_type对当前anchor进行标定
                    # y_is_box_valid为1时表示该anchor在训练中起作用，此时y_rpn_overlap的值表明了该anchor是正样本还是负样本
                    # y_is_box_valid为0时表示该anchor在训练中不起作用，此时y_rpn_overlap的值恒为0，即该anchor为中性样本
                    if anchor_type == 'neg':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif anchor_type == 'neutral':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 0
                    elif anchor_type == 'pos':
                        y_is_box_valid[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        y_rpn_overlap[jy, ix, anchor_ratio_idx + n_anchratios * anchor_size_idx] = 1
                        start = 4 * (anchor_ratio_idx + n_anchratios * anchor_size_idx)
                        y_rpn_regr[jy, ix, start:start + 4] = best_regr

    # 确保每个gt-box至少有一个正anchor （对于没有正anchor的gt-box，在与其相加的中性anchor里挑最好的）
    for idx in range(num_anchors_for_bbox.shape[0]):
        if num_anchors_for_bbox[idx] == 0:
            # 没有一个anchor与该gt-box相交
            if best_anchor_for_bbox[idx, 0] == -1:
                continue
            y_is_box_valid[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1],
                best_anchor_for_bbox[idx, 2] + n_anchratios * best_anchor_for_bbox[idx, 3]] = 1
            y_rpn_overlap[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1],
                best_anchor_for_bbox[idx, 2] + n_anchratios * best_anchor_for_bbox[idx, 3]] = 1
            start = 4 * (best_anchor_for_bbox[idx, 2] + n_anchratios * best_anchor_for_bbox[idx, 3])
            y_rpn_regr[
                best_anchor_for_bbox[idx, 0], best_anchor_for_bbox[idx, 1], start:start + 4] = best_dx_for_bbox[idx, :]

    # 变换数组维度，并在前面增加一维
    y_rpn_overlap = np.transpose(y_rpn_overlap, (2, 0, 1))  # shape(9,m,n)
    y_rpn_overlap = np.expand_dims(y_rpn_overlap, axis=0)  # shape(1,9,m,n)

    y_is_box_valid = np.transpose(y_is_box_valid, (2, 0, 1))  # shape(9,m,n)
    y_is_box_valid = np.expand_dims(y_is_box_valid, axis=0)  # shape(1,9,m,n)

    y_rpn_regr = np.transpose(y_rpn_regr, (2, 0, 1))  # shape(36,m,n)
    y_rpn_regr = np.expand_dims(y_rpn_regr, axis=0)  # shape(1,36,m,n)

    # 获得正负样本的位置
    pos_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 1, y_is_box_valid[0, :, :, :] == 1))
    neg_locs = np.where(np.logical_and(y_rpn_overlap[0, :, :, :] == 0, y_is_box_valid[0, :, :, :] == 1))

    num_regions = 256  # 设置正负样本数量的最大值
    num_pos = len(pos_locs[0])  # 正样本数量

    # 若正样本数量超过128，则随机抽取超出数量的样本将y_is_box_valid置位0
    if len(pos_locs[0]) > num_regions / 2:
        val_locs = random.sample(range(num_pos), num_pos - num_regions / 2)
        y_is_box_valid[0, pos_locs[0][val_locs], pos_locs[1][val_locs], pos_locs[2][val_locs]] = 0
        num_pos = num_regions / 2

    # 同上
    if len(neg_locs[0]) + num_pos > num_regions:
        val_locs = random.sample(range(len(neg_locs[0])), len(neg_locs[0]) - num_pos)
        y_is_box_valid[0, neg_locs[0][val_locs], neg_locs[1][val_locs], neg_locs[2][val_locs]] = 0

    # y_rpn_cls：shape(1,18,m,n), 将y_is_box_valid、y_rpn_overlap在第二个维度上连接
    # y_rpn_regr：shape(1,72,m,n), 将y_rpn_overlap在第二个维度重复4倍，再与y_rpn_regr在第二个维度上连接
    # 这里这么转换是因为在训练时计算loss需要这种格式
    y_rpn_cls = np.concatenate([y_is_box_valid, y_rpn_overlap], axis=1)
    y_rpn_regr = np.concatenate([np.repeat(y_rpn_overlap, 4, axis=1), y_rpn_regr], axis=1)

    return np.copy(y_rpn_cls), np.copy(y_rpn_regr)


def get_anchor_gt(all_img_data, C, img_length_calc_function, backend, mode='train'):
    """
    为rpn网络生成训练数据
    :param all_img_data: 列表，元素为字典，包含了原始图片信息
    :param C: 配置参数
    :param img_length_calc_function: 将图片尺寸转换到feature map尺寸的函数
    :param backend: Keras后端，'tf'
    :param mode: 'train'模式
    :return: 生成器，生成resize后的图片，标定好的anchor和回归系数，原始图片的信息
    """
    while True:
        '''
        if mode == 'train':
            random.shuffle(all_img_data)  # 打乱图片顺序
        '''
        for img_data in all_img_data:
            try:
                # 读入原始图片，并根据配置信息看是否做数据增强
                if mode == 'train':
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)
                else:
                    img_data_aug, x_img = data_augment.augment(img_data, C, augment=False)

                # 读取原始图片的宽和高
                width, height = (img_data['width'], img_data['height'])
                resized_width = width
                resized_height = height
                '''
                # 小数字目标太小，不做短边为600的resize了
                # 将原始图片resize到输入图片，短边为600
                resized_width, resized_height = get_new_img_size(width, height, C.im_size)
                x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
                '''
                # 得到标定好的anchor和回归系数
                # y_rpn_cls：shape(1,18,m,n), 第二维的前面9个数的值表明了哪些anchor在训练中起作用，后面9个数的值区分正负样本
                # y_rpn_regr：shape(1,72,m,n), 第二维的前面36个数是9个anchor是否为正负样本重复4次，后面36个数是对应的回归参数
                try:
                    y_rpn_cls, y_rpn_regr = calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)
                except:
                    continue

                # 对图片做处理，减去均值,像素归一化，调整维度顺序，增加维度
                # x_img = x_img[:, :, (2, 1, 0)]  # BGR -> RGB
                x_img = x_img.astype(np.float32)

                x_img[:, :, 0] -= C.img_channel_mean[0]
                x_img[:, :, 1] -= C.img_channel_mean[1]
                x_img[:, :, 2] -= C.img_channel_mean[2]

                x_img /= C.img_scaling_factor  # [??? 这个配置参数是什么意义]
                # x_img = np.transpose(x_img, (2, 0, 1)) # 顺时针翻转90度
                x_img = np.expand_dims(x_img, axis=0)

                # 把第二维后面的36个数乘以4（测试过程中会对应的除以4）[??? ]
                y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling

                if backend == 'tf':
                    # x_img = np.transpose(x_img, (0, 2, 3, 1))
                    y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                    y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug

            except Exception as e:
                print(e)
                continue
