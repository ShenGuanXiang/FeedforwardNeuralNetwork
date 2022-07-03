# 常见的几个激活函数及其导数
import numpy as np

# 输入输出格式如下：
# 激活函数
"""
:param z: ndarray: (<=batch_size, hidden_layer_dim, 1)
:return: ndarray: (<=batch_size, hidden_layer_dim, 1)
"""
# 激活函数的导数
"""
:param z: ndarray: (<=batch_size, hidden_layer_dim, 1)
:return: ndarray: (<=batch_size, hidden_layer_dim, hidden_layer_dim)
"""


# sigmoid&tanh的梯度消失问题后续可考虑批归一化来优化

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return np.array([np.diagflat(sigmoid(z[i]) * (1 - sigmoid(z[i]))) for i in range(z.shape[0])])


def tanh(z):
    return np.tanh(z)


def tanh_derivative(z):
    return np.array([np.diagflat(1 - np.power(tanh(z[i]), 2)) for i in range(z.shape[0])])


def relu(z):
    return np.maximum(0, z)


def relu_derivative(z):
    return np.array([np.diagflat((z[i] > 0).astype(int)) for i in range(z.shape[0])])


def softmax(z):
    """输入偏移，防止溢出"""
    shift_z = z - np.max(z)
    exps = np.exp(shift_z)
    return np.array([exps[i] / np.sum(exps[i]) for i in range(z.shape[0])])


def softmax_derivative(z):
    zs = softmax(z)
    return np.array([np.diagflat(zs[i]) - np.dot(zs[i], zs[i].T) for i in range(zs.shape[0])])
