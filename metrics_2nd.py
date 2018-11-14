import keras.backend as K
import tensorflow as tf
from tensorflow.contrib.metrics import streaming_mean_iou
import tensorflow as backend
import keras


def sample_accuracy(y_true, y_pred):
    """
    只管用于最后损失函数计算的采样样本的精度
    :param y_true:
    :param y_pred:
    :return:
    """
    labels = y_true[..., :-1]  # 去除标志位
    pixel_state = y_true[..., -1]  # 0 for ignore, 1 for object
    classification = y_pred

    #indices = backend.where(keras.backend.not_equal(pixel_state, 0))
    indices_1 = backend.where(keras.backend.not_equal(pixel_state, 0))  # 忽略大部分背景
    indices_2 = backend.where(keras.backend.not_equal(keras.backend.argmax(classification, -1), 15))  # 预测为数字的
    indices = backend.concat([indices_1, indices_2], 0)
    labels = backend.gather_nd(labels, indices)
    classification = backend.gather_nd(classification, indices)

    return keras.metrics.categorical_accuracy(labels,classification)


def accuracy(y_true, y_pred):
    """
    整个图片的语义分割精度
    :param y_true:
    :param y_pred:
    :return:
    """
    labels = y_true[..., :-1]  # 去除标志位
    classification = y_pred
    return keras.metrics.categorical_accuracy(labels,classification)

