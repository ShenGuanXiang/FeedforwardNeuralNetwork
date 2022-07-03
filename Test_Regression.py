# 测试神经网络：拟合曲线y=x**2
import FeedforwardNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

# 输入x,输出y,隐藏层只有一层，其节点数为16,其余超参数根据经验和实测进行了微调
myNN = FeedforwardNeuralNetwork.MyNeuralNetwork(input_dim=1, hidden_dims=list([16]), output_dim=1, batch_size=512
                                                , learning_rate=4e-4, l2_lambda=1e-3, activation_method="relu")

# 训练神经网络时均匀选取[-15,15]间的10000个点
train_input_data = np.linspace(start=-15, stop=15, num=10000)
train_output_data = train_input_data ** 2

# pretrain
myNN.train(train_input_data, train_output_data, train_times=150, visualize=False)
# train
myNN.train(train_input_data, train_output_data, train_times=2000, visualize=True)

# 测试神经网络时均匀选取[-15,15]间的900个点
test_input_data = np.linspace(start=-15, stop=15, num=900)
test_output_data = myNN.test(test_input_data, dump_dir="Results\\Regression_Test_Result.txt")

# 画图对比真实曲线和拟合曲线
plt.figure()
plt.title("regression_curve")
plt.plot(train_input_data, train_output_data, label="true")
plt.plot(test_input_data, test_output_data, label="predicted")
plt.legend()
plt.show()
