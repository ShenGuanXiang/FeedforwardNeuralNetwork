# 常见的几个损失函数及其导数

import numpy as np

# 输入输出格式如下：
# 损失函数
"""
:param y_predicted: ndarray: (<=batch_size, output_layer_dim, 1)
:param y_true: ndarray: (<=batch_size, output_layer_dim, 1)
:return: float
"""
# 损失函数的导数
"""
:param y_predicted: ndarray: (<=batch_size, output_layer_dim, 1)
:param y_true: ndarray: (<=batch_size, output_layer_dim, 1)
:return: ndarray: (<=batch_size, output_layer_dim, 1)
"""


# 正则化惩罚项并未添加到损失函数中，而是直接放在了参数更新处。

# Mean Squared Error
def mse(y_predicted, y_true):
    return 0.5 * np.sum(np.mean((y_predicted - y_true) ** 2, axis=0))


def mse_derivative(y_predicted, y_true):
    return (y_predicted - y_true).transpose((0, 2, 1))


# Cross Entropy
def ce(y_predicted, y_true):
    eps = np.finfo(float).eps  # 添加一个微小值可以防止负无限大(np.log(0))的发生。
    return -np.sum(np.mean(y_true * np.log(y_predicted + eps), axis=0))


def ce_derivative(y_predicted, y_true):
    eps = np.finfo(float).eps
    return (-y_true.astype(float) / (y_predicted + eps)).transpose((0, 2, 1))
