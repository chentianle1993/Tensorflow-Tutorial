# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

'''伪造数据'''
n_samples = 300
x_data = np.linspace(-1,1,n_samples) # shape = (300,)
x_data = x_data[np.newaxis,:] # 将一维数据转换为二维矩阵 shape = (1,300)
y_data = np.square(x_data) - 5 # shape = (1,300)

'''构造图'''
xs = tf.placeholder(tf.float32, [1, None])

num_midNeurons = 10 # 中间神经元个数
# layer 1
Weights_0_1 = tf.Variable(tf.random_normal([num_midNeurons, 1]))
biases_0_1 = tf.Variable(tf.zeros([num_midNeurons, 1]) + 0.1)
Wx_plus_b_0_1 = tf.matmul(Weights_0_1, xs) + biases_0_1 # shape = (10, 300)
layer1output = tf.nn.relu(Wx_plus_b_0_1) # shape = (10, 300)
# layer1output = Wx_plus_b_0_1 # shape = (10, 300)

# layer 2
Weights_1_2 = tf.Variable(tf.random_normal([num_midNeurons, 1]))
Wx_1_2_shape_10_300 = tf.multiply(Weights_1_2, layer1output) # type = tensor
Wx_1_2_shape_1_300 = tf.reduce_sum(Wx_1_2_shape_10_300, 0)
biases_1_2 = tf.Variable(tf.zeros([1, 1]) + 0.1) # shape = (1,1)
layer2output_shape_1_300 = Wx_1_2_shape_1_300 + biases_1_2


# for plot
# layer2output = o1 + ... + o10 + biases_1_2
#              = o1 + ... + o10 + biases_1_2 * ( o1 + ... + o10 ) / (o1 + ... + o10)
#              = o1 ( 1 + biases_1_2 /(o1 + ... +o10)) + ... + o10 ( 1 + biases_1_2 /(o1 + ... +o10) )
# [
#     [o1],
#     ...            = Wx_1_2_shape_10_300
#     [o10]
# ]
#
# [
#     [o1+...o10]    = Wx_1_2_shape_1_300
# ]
plotWeight = tf.add(1.0, tf.divide(biases_1_2, Wx_1_2_shape_1_300)) # shape = (1, 300)
plotY_Tensor = tf.multiply(Wx_1_2_shape_10_300, plotWeight) # shape = ( 10, 300 ) for 10 subplots


'''定义loss和train'''
ys = tf.placeholder(tf.float32, [1, None])

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - layer2output_shape_1_300),reduction_indices=[0]))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


'''运行图'''
sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion() # 让plt.show()执行之后不停止
plt.show()
fig = plt.figure(figsize=(10,1))
estimateLinesInSubplots = []
num_graphPaths = num_midNeurons

n_train = 100 # 训练次数
for i_train in range(0,n_train): # 训练次数
    # 命令行输出：初始/本次训练前的loss
    print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
    # 画图
    plotY_Ndarray = sess.run(plotY_Tensor, feed_dict={xs:x_data}) # shape = (10,300)
    for i_graphPath in range(0, num_graphPaths): # 在所有的子图画出预测结果
        currAx = fig.add_subplot(1, num_graphPaths, i_graphPath + 1)
        currAx.set_xlim(-1,1)
        currAx.set_ylim(-5,1)
        currAx.plot(x_data[0], y_data[0], "b-") # 原始数据
        currEstimateLine = currAx.plot(x_data[0], plotY_Ndarray[i_graphPath], 'r-')[0]
        estimateLinesInSubplots.append(currEstimateLine)
    plt.pause(1)
    # 训练
    resTrain_step = sess.run(train, feed_dict={xs:x_data, ys:y_data})
    # 清空图
    if i_train < n_train-1 : # 清空图，除了最后一次训练
        for i_graphPath in range(0, num_graphPaths):
            currAx = plt.subplot(1, num_graphPaths, i_graphPath + 1)
            currAx.clear()
