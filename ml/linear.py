#선형회귀
# 택시 주행거리와 택시비 예측
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# import numpy as np
# points = 200
# sets = []
# for i in range( points ) :
    # x = np.random.normal( 5, 5 ) + 15                # 독립변수 주행거리 
    # y = x * 1000 + (np.random.normal(0, 3)) * 1000   # 종속변수 택시비
    # sets.append( [x, y] )
# data = [ i[0] for i in sets ]
# label = [ i[1] for i in sets ]
# # y = Wx + b        W 기울기     b 절편
#
# W = tf.Variable( tf.random.uniform( shape=[1], minval=-1.0, maxval=1.0 ) )
# b = tf.Variable( tf.zeros( [1] ) )
# y = W * data + b
# loss = tf.reduce_mean( tf.square( y - label ) )      # 최소제곱법
# optimizer = tf.train.GradientDescentOptimizer( 0.001 ) 
    # # 0.1     오버슈팅            0.0001     스몰레이팅
# train = optimizer.minimize( loss )
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run( init )  
#
# import matplotlib.pyplot as plt
# for step in range( 10 ) :
    # sess.run( train )
    # print( step, sess.run( W ), sess.run( b ) )
    # print( step, sess.run( loss ) )
    # plt.plot( data, label, "ro" )
    # plt.plot( data, sess.run(W) * data + sess.run( b ) )
    # plt.xlabel( "distance" )
    # plt.ylabel( "price" )
    # plt.show()  
    
# 손글씨 분석
from tensorflow import keras
(train_data, train_label),(test_data, test_label) =\
    keras.datasets. mnist.load_data()

data = tf.placeholder(dtype=tf.float32, shape=[None,784])
label = tf.placeholder(dtype=tf.float32, shape=[None,10])
W = tf.Variable(tf.zeros([784,10]), dtype=tf.float32)
b = tf.Variable(tf.zeros([10]), dtype=tf.float32)

# y = softmax( Wx ) + b                              # 이항분류 - sigmoid  # 다항분류 - softmax
softmax = tf.nn.softmax(tf.matmul(data, W) + b )
y = tf.arg_max(softmax, 1)
log_y = y * tf.log(softmax)
tmp_cost = -tf.reduce_sum( log_y, reduction_indices=[1])
cost = tf.reduce_mean(tmp_cost)

train = tr.train.GradientDescentOptimizer(0.5).minimize(cost)
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(1000) :
    sess.run(train, feed_dict={data:train_data, label:train_label})
    if i%100==0 :
        Wout = sess.run(W, feed_dict={data:train_data})
        bout = sess.run( b, feed_dict={data:train_data})
        softmaxout = sess.run(tf.nn.softmax(tf.matmul( data ,W) + b), \
                              feed_dict={data:train_data})
        yout = sess.run( tf.argmax(softmax,1), feed_dict={data:train_data})
        log_yout = sess.run(y*tf.log(softmax), \
                            feed_dict = {data:train_data, label:train_label})
        tmp_costout = sess.run(-tf.reduce_sum(log_y, reduction_indices=[1], \
                                              feed_dict = {data:train_data, label:train_label}))
        costout = sess.run(tf.reduce_mean(tmp_cost), \
                           feed_dict = {data:train_data, label:train_label})
        
        
