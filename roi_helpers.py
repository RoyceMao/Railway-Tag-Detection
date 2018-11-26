import numpy as np
import math
import data_generators
import copy


def calc_iou(R, img_data, C, class_mapping):
    """
    生成后续分类网络的训练数据
    :param R: proposals
    :param img_data: 图片信息
    :param C: 配置参数
    :param class_mapping: 类别映射
    :return:
    """
    bboxes = img_data['bboxes']  # gt-box个数
    width, height = (img_data['width'], img_data['height'])  # 得到原始图片的宽和高
    # 将原始图片进行resize
    resized_width, resized_height = data_generators.get_new_img_size(width, height, C.im_size)

    gta = np.zeros((len(bboxes), 4))
    # 对原图中gt-box进行缩放，并映射到feature map上
    for bbox_num, bbox in enumerate(bboxes):
        gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width)) / C.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width)) / C.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height)) / C.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height)) / C.rpn_stride))

    x_roi = []
    y_class_num = []
    y_class_regr_coords = []
    y_class_regr_label = []
    IoUs = []  # for debugging only

    # 遍历每一个proposal
    for ix in range(R.shape[0]):
        (x1, y1, x2, y2) = R[ix, :]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        best_iou = 0.0
        best_bbox = -1

        # 遍历每一个gt-box
        for bbox_num in range(len(bboxes)):
            # 计算iou
            curr_iou = data_generators.iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]],
                                           [x1, y1, x2, y2])
            # 记录与当前proposal相交的最大iou值，以及相应的gt-box序号
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num

        # 如果当前proposal的最大iou值小于最小阈值，则丢弃
        if best_iou < C.classifier_min_overlap:
            continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])  # 存放左上角坐标和宽高
            IoUs.append(best_iou)

            # 如果当前proposal的最大iou值在最小阈值和最大阈值之间，则将其标为背景类
            if C.classifier_min_overlap <= best_iou < C.classifier_max_overlap:
                cls_name = 'bg'

            # 如果当前proposal大于等于最大阈值，则将其标为相应gt-box所属的类别，并计算回归系数
            elif C.classifier_max_overlap <= best_iou:
                cls_name = bboxes[best_bbox]['class']
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError

        # one-hot encoding, class_label是长度为21的列表, 在当前proposal所属类别的位置上置1
        class_num = class_mapping[cls_name]
        class_label = len(class_mapping) * [0]
        class_label[class_num] = 1

        y_class_num.append(copy.deepcopy(class_label))

        # coords、labels都是长度为80的列表（除了背景类外的20个类别乘以4）
        coords = [0] * 4 * (len(class_mapping) - 1)
        labels = [0] * 4 * (len(class_mapping) - 1)
        if cls_name != 'bg':
            label_pos = 4 * class_num
            sx, sy, sw, sh = C.classifier_regr_std  # [??? 不明白这个配置参数的意义]
            # 回归系数乘以C.classifier_regr_std，并存放到coords的对应位置上
            coords[label_pos:4 + label_pos] = [sx * tx, sy * ty, sw * tw, sh * th]
            # 将对应的labels四个位置全置1
            labels[label_pos:4 + label_pos] = [1, 1, 1, 1]

            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    # 存放了 best_iou ≥ C.classifier_min_overlap 的proposal的坐标信息[x1, y1, w, h]，shape(selected_boxes_num,4)
    X = np.array(x_roi)
    # 每个proposal包含的物体类别，shape(selected_boxes_num, 21)
    Y1 = np.array(y_class_num)
    # shape(selected_boxes_num,160)，前面80个对应物体类别，后面80个数对应回归系数乘以C.classifier_regr_std
    Y2 = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis=1)

    # 增加一个维度并返回
    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs


def apply_regr(x, y, w, h, tx, ty, tw, th):
    try:
        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy
        w1 = math.exp(tw) * w
        h1 = math.exp(th) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.
        x1 = int(round(x1))
        y1 = int(round(y1))
        w1 = int(round(w1))
        h1 = int(round(h1))

        return x1, y1, w1, h1

    except ValueError:
        return x, y, w, h
    except OverflowError:
        return x, y, w, h
    except Exception as e:
        print(e)
        return x, y, w, h


def apply_regr_np(X, T):
    """
    修正边框位置
    :param X:
    :param T:
    :return:
    """
    try:
        x = X[0, :, :]
        y = X[1, :, :]
        w = X[2, :, :]
        h = X[3, :, :]

        tx = T[0, :, :]
        ty = T[1, :, :]
        tw = T[2, :, :]
        th = T[3, :, :]

        cx = x + w / 2.
        cy = y + h / 2.
        cx1 = tx * w + cx
        cy1 = ty * h + cy

        w1 = np.exp(tw.astype(np.float64)) * w
        h1 = np.exp(th.astype(np.float64)) * h
        x1 = cx1 - w1 / 2.
        y1 = cy1 - h1 / 2.

        x1 = np.round(x1)
        y1 = np.round(y1)
        w1 = np.round(w1)
        h1 = np.round(h1)
        return np.stack([x1, y1, w1, h1])
    except Exception as e:
        print(e)
        return X


def non_max_suppression_fast(boxes, overlap_thresh=0.9, max_boxes=300):
    """
    使用nms过滤多余的边框
    :param boxes: 数组, shape(m*n*9,5), 存放了所有anchor的坐标以及预测分数
    :param overlap_thresh: 最大阈值
    :param max_boxes: 最多生产多少个proposals
    :return: 返回按得分降序排列的proposals
    """
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes)
    # 提取边框左上角和右下角坐标
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []  # 用于存放被选取的边框索引
    area = (x2 - x1) * (y2 - y1)  # 计算各个边框的面积
    indexes = np.argsort([i[-1] for i in boxes])  # 根据预测分数对索引进行排序 (升序)

    while len(indexes) > 0:
        last = len(indexes) - 1
        i = indexes[last]  # 获取最高分数对应的索引
        pick.append(i)

        # 计算其他所有边框与其的交集
        xx1_int = np.maximum(x1[i], x1[indexes[:last]])
        yy1_int = np.maximum(y1[i], y1[indexes[:last]])
        xx2_int = np.minimum(x2[i], x2[indexes[:last]])
        yy2_int = np.minimum(y2[i], y2[indexes[:last]])
        # 计算宽和高
        ww_int = np.maximum(0, xx2_int - xx1_int)
        hh_int = np.maximum(0, yy2_int - yy1_int)
        # 计算面积
        area_int = ww_int * hh_int
        # 计算交集面积
        area_union = area[i] + area[indexes[:last]] - area_int

        # 计算iou
        overlap = area_int / (area_union + 1e-6)

        # 将iou大于最大阈值的边框删除
        indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

        if len(pick) >= max_boxes:
            break

    return boxes[pick]


def rpn_to_roi(rpn_layer, regr_layer, cfg, dim_ordering, use_regr=True, max_boxes=300, overlap_thresh=0.9):
    """
    综合rpn网络两个分支的预测结果，得到最终的proposals
    :param rpn_layer: 数组, shape(1,m,n,9), 存放了每个anchor的标定信息（预测分数）
    :param regr_layer: 数组, shape(1,m,n,36), 存放了每个anchor的回归系数
    :param cfg: 配置信息
    :param dim_ordering: 维度顺序
    :param use_regr: 是否调整边框位置
    :param max_boxes: 最多生成多少个proposals
    :param overlap_thresh: iou最大阈值
    :return: proposals
    """
    # 因为在训练过程中乘以了4，这里要对应除以4
    regr_layer = regr_layer / cfg.std_scaling

    anchor_sizes = cfg.anchor_box_scales  # [128, 256, 512]
    anchor_ratios = cfg.anchor_box_ratios  # [[1,1], [1,2], [2,1]]

    # assert rpn_layer.shape[0] == 1

    if dim_ordering == 'th':
        (rows, cols) = rpn_layer.shape[2:]
    else:
        (rows, cols) = rpn_layer.shape[1:3]

    # 对于tf后端，A的维度为(4,m,n,9)
    if dim_ordering == 'tf':
        A = np.zeros((4, rpn_layer.shape[1], rpn_layer.shape[2], rpn_layer.shape[3]))
    else:
        A = np.zeros((4, rpn_layer.shape[2], rpn_layer.shape[3], rpn_layer.shape[1]))

    curr_layer = 0  # curr_layer初始值为0，在下面的for循环中递增，代表9种不同形状的anchor
    for anchor_size in anchor_sizes:
        for anchor_ratio in anchor_ratios:
            # 计算将anchor映射到feature map上相应的宽和高
            anchor_x = (anchor_size * anchor_ratio[0]) / cfg.rpn_stride
            anchor_y = (anchor_size * anchor_ratio[1]) / cfg.rpn_stride
            if dim_ordering == 'th':
                regr = regr_layer[0, 4 * curr_layer:4 * curr_layer + 4, :, :]
            else:
                # 得到当前同形状的anchor的回归系数, 并调整维度顺序(4,m,n)
                regr = regr_layer[0, :, :, 4 * curr_layer:4 * curr_layer + 4]
                regr = np.transpose(regr, (2, 0, 1))

            # 生成两个形状为(m,n)的矩阵，X的每一列的值都相同，为当前列的index，Y的每一行的值都相同，为当前行的index
            X, Y = np.meshgrid(np.arange(cols), np.arange(rows))

            # 下面是计算映射到feature map上的anchor的左上角坐标，以及宽高
            A[0, :, :, curr_layer] = X - anchor_x / 2
            A[1, :, :, curr_layer] = Y - anchor_y / 2
            A[2, :, :, curr_layer] = anchor_x
            A[3, :, :, curr_layer] = anchor_y

            # 根据回归系数修正边框位置
            if use_regr:
                A[:, :, :, curr_layer] = apply_regr_np(A[:, :, :, curr_layer], regr)

            # 让每个anchor的宽和高都不小于1
            A[2, :, :, curr_layer] = np.maximum(1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.maximum(1, A[3, :, :, curr_layer])
            # 计算anchor的右下角坐标
            A[2, :, :, curr_layer] += A[0, :, :, curr_layer]
            A[3, :, :, curr_layer] += A[1, :, :, curr_layer]
            # 规定anchor的左上角和右下角坐标不能超过feature map的边界
            A[0, :, :, curr_layer] = np.maximum(0, A[0, :, :, curr_layer])
            A[1, :, :, curr_layer] = np.maximum(0, A[1, :, :, curr_layer])
            A[2, :, :, curr_layer] = np.minimum(cols - 1, A[2, :, :, curr_layer])
            A[3, :, :, curr_layer] = np.minimum(rows - 1, A[3, :, :, curr_layer])

            curr_layer += 1

    # 将每个anchor的坐标存入all_boxes中，shape(m*n*9,4)
    all_boxes = np.reshape(A.transpose((0, 3, 1, 2)), (4, -1)).transpose((1, 0))
    # 将每个anchor的预测得分存入all_probs中，shape(m*n*9,)
    all_probs = rpn_layer.transpose((0, 3, 1, 2)).reshape((-1))

    # 当左上角的坐标点超过了右上角的坐标，说明产生了错误，将这些anchor删除
    x1 = all_boxes[:, 0]
    y1 = all_boxes[:, 1]
    x2 = all_boxes[:, 2]
    y2 = all_boxes[:, 3]
    ids = np.where((x1 - x2 >= 0) | (y1 - y2 >= 0))
    all_boxes = np.delete(all_boxes, ids, 0)
    all_probs = np.delete(all_probs, ids, 0)

    # 水平合并all_boxes和all_probs，此时维度变为(m*n*9,5)
    all_boxes = np.hstack((all_boxes, np.array([[p] for p in all_probs])))
    # 进行nms过滤多余的边框
    result = non_max_suppression_fast(all_boxes, overlap_thresh=overlap_thresh, max_boxes=max_boxes)
    # 忽略最后一列的概率值，此时维度又变回(m*n*9,4)
    result = result[:, 0: -1]
    return result
