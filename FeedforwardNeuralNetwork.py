# 全连接的前馈神经网络(feedforward neural network)
# 采用反向传播算法(back propagation)中的梯度下降算法进行训练，添加L2正则化项，实现批量训练
import copy

import numpy as np
import ActivationMethods
import LossMethods
import matplotlib.pyplot as plt


def mydot(a: np.ndarray, b: np.ndarray):
    """
    重写numpy矩阵乘法，如果输入矩阵是三维的，会取该矩阵沿axis=0得到的每一个矩阵进行乘法，得到的结果是一个三维矩阵
    这个方法应用于正&反向传播，有助于实现批量训练
    """
    if a.ndim <= 2 and b.ndim <= 2:
        return np.dot(a, b)
    if a.ndim == 2 and b.ndim == 3:
        return np.array([np.dot(a, b[i]) for i in range(b.shape[0])])
    if a.ndim == 3 and b.ndim == 2:
        return np.array([np.dot(a[i], b) for i in range(a.shape[0])])
    if a.ndim == 3 and b.ndim == 3:
        assert a.shape[0] == b.shape[0]
        return np.array([np.dot(a[i], b[i]) for i in range(a.shape[0])])
    raise ValueError


class MyNeuralNetwork:

    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, activation_method="relu", loss_method="mse",
                 l2_lambda=1e-5, learning_rate=1e-4, batch_size=1):
        """
        :param input_dim: 输入层节点数
        :param hidden_dims: 各个隐藏层的节点数组成的列表
        :param output_dim: 输出层节点数
        :param activation_method: {"sigmoid","tanh", "relu", "softmax"} 隐藏层采用的激活函数。方便起见，各个隐藏层采用相同的激活函数且输出层不采用激活函数
        :param loss_method: {"mse","ce"} 损失函数
        :param l2_lambda: L2正则化惩罚项系数
        :param learning_rate: 学习率
        :param batch_size: 每轮训练取样数量
        """
        self.batch_size = batch_size
        self.depth = len(hidden_dims)  # 用隐藏层数目表示神经网络的深度
        # 初始化各样本各层节点(分配空间)，直接置零
        self.input_layer = np.zeros((batch_size, input_dim, 1))
        self.hidden_layers_in = list()
        self.hidden_layers_out = list()
        for i in range(self.depth):
            self.hidden_layers_in.append(np.zeros((batch_size, hidden_dims[i], 1)))
            self.hidden_layers_out.append(np.zeros((batch_size, hidden_dims[i], 1)))
        self.output_layer = np.zeros((batch_size, output_dim, 1))
        # 初始化网络参数，各层权重矩阵w采用标准正态分布的初始化方式，偏置矩阵b直接置零
        self.w = list([0 for i in range(self.depth + 1)])
        self.b = list([0 for i in range(self.depth + 1)])
        self.w[0] = np.random.randn(hidden_dims[0], input_dim)
        self.b[0] = np.zeros((hidden_dims[0], 1))
        for i in range(1, self.depth):
            self.w[i] = np.random.randn(hidden_dims[i], hidden_dims[i - 1])
            self.b[i] = np.zeros((hidden_dims[i], 1))
        self.w[self.depth] = np.random.randn(output_dim, hidden_dims[self.depth - 1])
        self.b[self.depth] = np.zeros((output_dim, 1))
        # 初始化激活函数、损失函数及它们的导函数的名称，后续通过函数名字符串直接调用
        self.activation_method = "ActivationMethods." + activation_method
        self.activation_derivative_method = self.activation_method + "_derivative"
        self.loss_method = "LossMethods." + loss_method
        self.loss_derivative_method = self.loss_method + "_derivative"
        # 初始化惩罚项系数，方便起见，惩罚项系数不随训练次数变化
        self.l2_lambda = l2_lambda
        # 初始化学习率，方便起见，也不变
        self.learning_rate = learning_rate

    def feedforward(self, x: np.ndarray):
        """
        批量前向传播一遍
        :param x: 批量获取的样本的输入数据  ndarray:(<=batch_size, input_layer_dim, 1)
        :return: 这些样本输入经过神经网络后的输出数据  ndarray:(<=batch_size, output_layer_dim, 1)
        """
        # 在下面的代码中，各个样本的节点数据是同时计算的
        # 输入层->隐藏层
        self.input_layer[np.arange(x.shape[0]), :, :] = copy.deepcopy(x)
        self.hidden_layers_in[0] = mydot(self.w[0], self.input_layer) + self.b[0]
        self.hidden_layers_out[0] = eval(self.activation_method)(self.hidden_layers_in[0])
        # 隐藏层间传播
        for i in range(1, self.depth):
            self.hidden_layers_in[i] = mydot(self.w[i], self.hidden_layers_out[i - 1]) + self.b[i]
            self.hidden_layers_out[i] = eval(self.activation_method)(self.hidden_layers_in[i])
        # 隐藏层->输出层
        self.output_layer = mydot(self.w[self.depth], self.hidden_layers_out[self.depth - 1]) + self.b[self.depth]
        return self.output_layer[np.arange(x.shape[0]), :, :]

    def backpropagation(self, y_true: np.ndarray):
        """
        批量反向传播一遍，期间会更新参数
        :param y_true: 真实值， ndarray:(<=batch_size, output_layer_dim, 1)
        """
        # 在下面的代码中，各个样本同时反向传播并取平均来更新参数
        # 输出层->隐藏层
        grads = eval(self.loss_derivative_method)(self.output_layer, y_true)  # 首先是损失函数求导
        old_w = copy.deepcopy(self.w[self.depth])
        dws = mydot(self.hidden_layers_out[self.depth - 1], grads).transpose((0, 2, 1))
        self.w[-1] -= self.learning_rate * (np.mean(dws, axis=0) + self.l2_lambda * self.w[-1])
        self.b[-1] -= self.learning_rate * (np.mean(grads.transpose((0, 2, 1)), axis=0) + self.l2_lambda * self.b[-1])
        grads = mydot(grads, old_w)  # 层间全连接求导
        # 隐藏层间传播
        for i in range(self.depth - 1, 0, -1):
            grads = mydot(grads, eval(self.activation_derivative_method)(self.hidden_layers_in[i]))  # 激活函数求导
            old_w = copy.deepcopy(self.w[i])  # old_w用来保存该处旧的权重矩阵，因为继续反向传播时需要用到wi，但在这之前wi被更新了
            dws = mydot(self.hidden_layers_out[i - 1], grads).transpose((0, 2, 1))  # 计算各个样本的损失函数对该处权重矩阵的导数矩阵
            self.w[i] -= self.learning_rate * (np.mean(dws, axis=0) + self.l2_lambda * self.w[i])
            self.b[i] -= self.learning_rate * (np.mean(grads.transpose((0, 2, 1)), axis=0) + self.l2_lambda * self.b[i])
            grads = mydot(grads, old_w)  # 层间全连接求导
        # 隐藏层->输入层
        grads = mydot(grads, eval(self.activation_derivative_method)(self.hidden_layers_in[0]))
        dws = mydot(self.input_layer, grads).transpose((0, 2, 1))
        self.w[0] -= self.learning_rate * (np.mean(dws, axis=0) + self.l2_lambda * self.w[0])
        self.b[0] -= self.learning_rate * (np.mean(grads.transpose((0, 2, 1)), axis=0) + self.l2_lambda * self.b[0])

    def train(self, input_data, output_data, train_times=100, visualize=False):
        """
        :param input_data: 输入数据
        :param output_data: 正确结果
        :param train_times: 训练轮数
        :param visualize: 是否可视化训练过程
        """
        x = np.array(input_data).reshape((-1, self.input_layer.shape[1], self.input_layer.shape[2]))
        y_true = np.array(output_data).reshape((-1, self.output_layer.shape[1], self.output_layer.shape[2]))
        assert x.shape[0] == y_true.shape[0]
        # 训练
        loss_list = list()
        for epoch in range(train_times):
            # 随机取样
            index = np.random.choice(np.arange(input_data.shape[0]), size=min(self.batch_size, input_data.shape[0]))
            x_batch = x[index, :, :]
            y_true_batch = y_true[index, :, :]
            # 前向传播、计算损失、反向传播
            self.feedforward(x_batch)
            loss_list.append(eval(self.loss_method)(self.output_layer, y_true_batch))
            self.backpropagation(y_true_batch)
        if visualize:
            epochs = np.arange(train_times)
            losses = np.array(loss_list)
            plt.title("loss")
            plt.plot(epochs, losses)
            plt.show()

    def test(self, input_data, dump_dir=None):
        """
        :param input_data: 输入数据
        :param dump_dir: 保存输入数据和测试结果
        :return: 输入数据经过神经网络的测试结果
        """
        x = np.array(input_data).reshape((-1, self.input_layer.shape[1], self.input_layer.shape[2]))
        y = np.zeros((x.shape[0], self.output_layer.shape[1], 1))
        for i in range(0, x.shape[0], self.batch_size):
            index = np.arange(i, min(i + self.batch_size, x.shape[0]))
            x_batch = x[index, :, :]
            y_batch = self.feedforward(x_batch)
            y[index, :, :] = y_batch
        if dump_dir is not None:
            with open(dump_dir, 'w') as f:
                f.truncate()
                print("x=\n", file=f)
                print(x, file=f)
                print("y=\n", file=f)
                print(y, file=f)
        return y.squeeze()
