# 퍼셉트론
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# y = W * x + b
dataset = np.genfromtxt( "x_square.txt", dtype=np.float32 )
# print( dataset.shape )        # (1999, 2)
train = dataset[:, 0]
label = dataset[:, 1]
# print( train.shape )          # (1999,)

train = np.reshape( train, [1, -1] )
label = np.reshape( label, [1, -1] )
# print( train.shape )          # (1, 1999)

# 입력층 설계
x = tf.placeholder( dtype=tf.float32, shape=[1, None] )
y = tf.placeholder( dtype=tf.float32, shape=[1, None] )

# 은닉층 설계
number = 10
W = tf.Variable( tf.random_normal( [number, 1] ) )   # 가중치
b = tf.Variable( tf.random_normal( [number, 1] ) )   # 편차
    # (10, 1) * (1, 1999)       (10, 1999)
hidden_layer1 = tf.nn.sigmoid( tf.matmul( W, x ) + b )

a_middle = tf.Variable( tf.random_normal( [number, number] ) )
b_middle = tf.Variable( tf.random_normal( [number, 1] ) )
hidden_layer2 = tf.nn.sigmoid( \
                    tf.matmul( a_middle, hidden_layer1) + b_middle )

# 출력층 설계
out = tf.Variable( tf.random_normal( [1, number] ) )
b_out = tf.Variable( tf.random_normal( [1, 1] ) )
y_out = tf.matmul( out, hidden_layer2 ) + b_out
    # (1, 10) * ( 10, 1999 )    ( 1, 1999 )

# 비용계산
cost = tf.nn.l2_loss( y_out - label )
optimizer = tf.train.AdamOptimizer( 0.1 )
do_train = optimizer.minimize( cost )
init = tf.global_variables_initializer()
with tf.Session() as sess :
    sess.run( init )
    for i in range( 1000 ) :
        sess.run( do_train, feed_dict={x:train, y:label} )
        # 학습
        test_temp = np.linspace( 0, 20, 50 )   # 0~20 50개 데이터 반환
        test_data = [test_temp]
        test_label = sess.run( y_out, feed_dict={x:test_data} )
                
import matplotlib.pyplot as plt
plt.plot( train, label, "ro", alpha=0.5 )
plt.plot( test_data, test_label, "b^", alpha=1 )
plt.show()        

