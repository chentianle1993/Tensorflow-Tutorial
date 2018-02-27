# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

'''伪造数据'''
x_data = np.linspace(-1,1,300) # shape = (300,)
x_data = x_data[np.newaxis,:] # 将一维数据转换为二维矩阵 shape = (1,300)
y_data = np.square(x_data) - 5 # shape = (1,300) value belongs to [-5, -4]

'''构造图'''
xs = tf.placeholder(tf.float32, [1, None])

num_midNeurons = 10 # 中间神经元个数

# layer 1
Weights_0_1 = tf.Variable(tf.random_normal([num_midNeurons, 1]))
biases_0_1 = tf.Variable(tf.zeros([num_midNeurons, 1]) + 0.1)
Wx_plus_b_0_1 = tf.matmul(Weights_0_1, xs) + biases_0_1 # type = (10, 300)
layer1output = tf.nn.relu(Wx_plus_b_0_1) # type = (10, 300)

# layer 2
Weights_1_2 = tf.Variable(tf.random_normal([1, num_midNeurons]))
biases_1_2 = tf.Variable(tf.zeros([1, 1]) + 0.1) # type = (1,1)
Wx_plus_b_1_2 = tf.matmul(Weights_1_2, layer1output) + biases_1_2
layer2output = Wx_plus_b_1_2 # type = (1,300)


'''定义loss和train'''
ys = tf.placeholder(tf.float32, [1, None])
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - layer2output),reduction_indices=[0]))
train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


'''运行图'''
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i_train in range(1,10): # 训练次数
    # 先算loss
    print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
    # print(sess.run(Weights_0_1, feed_dict={xs:x_data, ys:y_data}))
    # 再改进
    resTrain_step = sess.run(train, feed_dict={xs:x_data, ys:y_data})

