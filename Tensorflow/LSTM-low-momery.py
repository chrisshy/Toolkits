import numpy as np
import matplotlib.pyplot as plt
#import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
#tf.global_variables_initializer
import pandas as pd
import math
class MyLSTM:
    def __init__(self,iteration=5,rnn_unit=8,lstm_layers=1,output_size=1,lr=0.01,timestep = 20,validation=0.2,batch_size=100,forget_bias=0.3):
        """
        parameters here:  
        rnn_unit: LSTM单元(一层神经网络)中的中神经元的个数
        lstm_layers: LSTM单元个数，双层、多层等
        output_size: 输出神经元个数（预测值），回归的话应为1
        time_step: 时间步，即每做一次预测需要先输入的时刻数据数量
        validation: 切割训练集和validation数据集，提高预测鲁棒性
        batch_size: 每一批次训练多少个样例
        """
        self.rnn_unit = rnn_unit
        self.lstm_layers = lstm_layers
        self.output_size = output_size
        self.lr = lr
        self.timestep = timestep
        self.validation = validation
        self.batch_size = batch_size
        self.forget_bias = forget_bias
        self.iteration = iteration


    def fit(self,data):
        n1 = data.shape[1] - 1  # 因为最后一位为label
        n2 = data.shape[0]
        print(f'特征维度:{n1}, 观察值:{n2}')
        input_size = n1  # 输入神经元个数，特征维度
        train_end_index = int(n2 * (1-self.validation))  # 向下取整
        print('train_begin_index', 0)
        print('train_end_index', train_end_index)
        # ——————————————————定义神经网络变量——————————————————
        # 输入层、输出层权重、偏置、dropout参数
        # 随机产生 w,b
        weights = {
            'in': tf.Variable(tf.random.normal([input_size, self.rnn_unit])),#input_size行，每一行有rnn_unit列
            'out': tf.Variable(tf.random.normal([self.rnn_unit, 1]))#rnn_unit 行，1列，每一行都是一维向量
        }
        biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[self.rnn_unit, ])),  #rnn_unit的列向量
            'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
        }
        keep_prob = tf.placeholder(tf.float16, name='keep_prob')  # dropout 防止过拟合

        # ——————————————————定义神经网络——————————————————
        def lstmCell():
            basicLstm = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_unit, forget_bias=self.forget_bias, state_is_tuple=True)
            # dropout 未使用
            drop = tf.nn.rnn_cell.DropoutWrapper(basicLstm, output_keep_prob=keep_prob)
            return basicLstm

        def lstm(X):  # 参数：输入网络批次数目
            batch_size = tf.shape(X)[0]  # 将矩阵的维度输出为一个维度矩阵
            time_step = tf.shape(X)[1]
            w_in = weights['in']
            b_in = biases['in']
            input = tf.reshape(X, [-1, input_size])
            input_rnn = tf.matmul(input, w_in) + b_in
            input_rnn = tf.reshape(input_rnn, [-1, time_step, self.rnn_unit])
            print('input_rnn', input_rnn)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstmCell() for i in range(self.lstm_layers)])
            init_state = cell.zero_state(batch_size, dtype=tf.float32)

            # 输出门
            w_out = weights['out']
            b_out = biases['out']
            # output_rnn是最后一层每个step的输出,final_states是每一层的最后那个step的输出
            output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)
            output = tf.reshape(output_rnn, [-1, self.rnn_unit])
            # 输出值，同时作为下一层输入门的输入
            pred = tf.matmul(output, w_out) + b_out
            return pred, final_states

        # ————————————————训练模型————————————————————

        def train_lstm(batch_size=self.batch_size, time_step=self.timestep, train_begin=0, train_end=train_end_index):
            X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])
            Y = tf.placeholder(tf.float32, shape=[None, time_step, self.output_size])

            batch_index = []
            data_train = data.iloc[train_begin:train_end]
#             normalized_train_data = (data_train - np.mean(data_train, axis=0)) / np.std(data_train, axis=0)  # 标准化

            
            for i in range(data_train.shape[0] - time_step):
                if i % batch_size == 0:
                    # 开始位置
                    batch_index.append(i)
            # 结束位置
            batch_index.append(data_train.shape[0] - time_step)

            # 用tf.variable_scope来定义重复利用,LSTM会经常用到
            with tf.variable_scope("sec_lstm"):
                pred, state_ = lstm(X)  # pred输出值，state_是每一层的最后那个step的输出
            print('pred,state_', pred, state_)
            loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))

            train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

            with tf.Session() as sess:
                # 初始化
                sess.run(tf.global_variables_initializer())
                theloss = []
                # 迭代次数
                print(len(batch_index))
                for i in range(self.iteration):
                    for step in range(len(batch_index) - 1):
                    #for step in range(len(batch_index)):
                        batch_data_begin = batch_index[step]
                        batch_data_end = batch_index[step+1]
                        train_x_this_batch = []
                        train_y_this_batch = []
                        for item in range(batch_data_begin,batch_data_end):
                            train_x_this_batch.append(np.array(data_train.iloc[item:item + time_step, :n1]))
                            train_y_this_batch.append(np.array(data_train.iloc[item:item + time_step, n1])[:, np.newaxis])

                        # sess.run(b, feed_dict = replace_dict)
                        state_, loss_ = sess.run([train_op, loss],
                                                feed_dict={X: train_x_this_batch,
                                                            Y: train_y_this_batch,
                                                            keep_prob: 0.5})
                        print(step)
                    print("Number of iterations:", i, " loss:", loss_)
                    theloss.append(loss_)
                print("model_save: ", saver.save(sess, 'model_save2\\modle.ckpt'))
                print("The train has finished")
            return theloss

        theloss = train_lstm()

        # ————————————————预测模型————————————————————
        def prediction(time_step=self.timestep):
            
            X = tf.placeholder(tf.float32, shape=[None, time_step, input_size])

            test_begin=train_end_index + 1
            data_test = data.iloc[test_begin:]
            # 标准化(归一化）
#             normalized_test_data = (data_test - mean) / std
            test_size = (data_test.shape[0] + time_step - 1) // time_step
            print('test_size$$$$$$$$$$$$$$', test_size)

            # 用tf.variable_scope来定义重复利用,LSTM会经常用到
            with tf.variable_scope("sec_lstm", reuse=tf.AUTO_REUSE):
                pred, state_ = lstm(X)
            saver = tf.train.Saver(tf.global_variables())
            with tf.Session() as sess:
                # 参数恢复（读取已存在模型）
                module_file = tf.train.latest_checkpoint('model_save2')
                saver.restore(sess, module_file)
                test_predict = []
                #for step in range(len(test_x) - 1):
                for step in range(test_size):
                    if i < test_size-1:
                        x = np.array(data_test.iloc[i * time_step:(i + 1) * time_step, :n1])
                        predict = sess.run(pred, feed_dict={X: [x], keep_prob: 1})
                        predict = predict.reshape((-1))
                        test_predict.extend(predict)  # 把predict的内容添加到列表
                    else:
                        x = np.array(data_test.iloc[(test_size-1) * time_step:, :n1])
                        predict = sess.run(pred, feed_dict={X: [x], keep_prob: 1})
                        predict = predict.reshape((-1))
                        test_predict.extend(predict)  # 把predict的内容添加到列表

                test_predict = np.array(test_predict)

                
                # MAE
                mae = np.average(np.abs(test_predict - data_test.iloc[:,n1]))
                print("预测的MAE:",mae)

                print(theloss)
                plt.figure()
                plt.plot(list(range(len(theloss))), theloss, color='b', )
                plt.xlabel('times', fontsize=14)
                plt.ylabel('loss valuet', fontsize=14)
                plt.title('loss-----blue', fontsize=10)
                plt.show()
                # 以折线图表示预测结果
                plt.figure()
                plt.plot(list(range(len(test_predict))), test_predict, color='b', )
                plt.plot(list(range(len(data_test.iloc[:,n1]))), data_test.iloc[:,n1], color='r')
                plt.xlabel('time value/day', fontsize=14)
                plt.ylabel('volatility value/point', fontsize=14)
                plt.title('predict-----blue,real-----red', fontsize=10)
                plt.show()
        prediction()