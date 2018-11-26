from keras import backend as K
from keras.objectives import categorical_crossentropy

if K.image_dim_ordering() == 'tf':
    import tensorflow as tf

lambda_rpn_regr = 1.0
lambda_rpn_class = 1.0

lambda_cls_regr = 1.0
lambda_cls_class = 1.0

epsilon = 1e-4


def rpn_loss_regr(y_true, y_pred):
    """
    :param y_true: ground truth, shape(1,m,n,72)
    :param y_pred: 预测的回归系数, shape(1,m,n,36)
    :return:
    """
    num_anchors = 9
    if K.image_dim_ordering() == 'th':
        x = y_true[:, 4 * num_anchors:, :, :] - y_pred
        x_abs = K.abs(x)
        x_bool = K.less_equal(x_abs, 1.0)
        return lambda_rpn_regr * K.sum(
            y_true[:, :4 * num_anchors, :, :] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :4 * num_anchors, :, :])
    else:
        x = y_true[:, :, :, 4 * num_anchors:] - y_pred  # 计算预测值与ground truth的差值
        x_abs = K.abs(x)  # 取绝对值
        x_bool = K.cast(K.less_equal(x_abs, 1.0), tf.float32)  # 判断是否小于1，并将bool转换为float

        # 前面乘以 y_true[:, :, :, :4 * num_anchors] 是因为只对正样本求回归误差
        # 后面的 K.sum(epsilon + y_true[:, :, :, :4 * num_anchors]) 代表正样本个数，加上epsilon是防止除以0
        return lambda_rpn_regr * K.sum(
            y_true[:, :, :, :4 * num_anchors] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :, :4 * num_anchors])



def rpn_loss_cls(y_true, y_pred):
    """
    :param y_true: ground truth, shape(1,m,n,18)
    :param y_pred: 预测类别得分, shape(1,m,n,9)
    :return:
    """
    num_anchors = 9
    if K.image_dim_ordering() == 'tf':
        # 乘以 K.sum(y_true[:, :, :, :num_anchors] 是为了只计算正负样本的交叉熵
        return lambda_rpn_class * K.sum(y_true[:, :, :, :num_anchors] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, :, :, num_anchors:])) / K.sum(epsilon + y_true[:, :, :, :num_anchors])
    else:
        return lambda_rpn_class * K.sum(y_true[:, :num_anchors, :, :] * K.binary_crossentropy(y_pred[:, :, :, :], y_true[:, num_anchors:, :, :])) / K.sum(epsilon + y_true[:, :num_anchors, :, :])


def class_loss_cls(y_true, y_pred):
    """
    计算分类网络的分类误差
    :param y_true: ground truth, shape(1,selected boxes,21)
    :param y_pred: 预测类别得分, shape(1,selected boxes,21)
    :return:
    """
    return lambda_cls_class * K.sum(y_true[:, :, 0] * K.categorical_crossentropy(y_true[:, :, 1:], y_pred[:, :, :])) / K.sum(epsilon + y_true[:, :, 0])


def class_loss_regr(y_true, y_pred):
    """
    计算分类网络的回归误差
    :param y_true: ground truth, shape(1,selected boxes,160)
    :param y_pred: 预测的回归系数, shape(1,selected boxes,80)
    :return:
    """
    num_classes = 10
    x = y_true[:, :, 4*num_classes:] - y_pred
    x_abs = K.abs(x)
    x_bool = K.cast(K.less_equal(x_abs, 1.0), 'float32')
    return lambda_cls_regr * K.sum(y_true[:, :, :4*num_classes] * (x_bool * (0.5 * x * x) + (1 - x_bool) * (x_abs - 0.5))) / K.sum(epsilon + y_true[:, :, :4*num_classes])


