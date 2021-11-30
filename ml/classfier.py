# KNN
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
data = np.genfromtxt( "label_dots.txt", dtype=np.float32 )
test = np.genfromtxt( "test_dots.txt", dtype=np.float32 )
train_data = data[:, :2]
train_label = data[:, 2:5]
test_data = test[:, :2]
test_label = test[:, 2:5]
train = tf.constant( train_data )
label = tf.constant( train_label )

print( train_data.shape )       # (899, 2)
print( train_label.shape )      # (899, 3)

x = tf.placeholder( dtype=tf.float32, shape=[2] )
dtrain = tf.reshape( train, [1, -1, 2] )    # [1, 899, 2]
dx = tf.reshape( x, [1, -1, 2] )            # [1, 1, 2]

dist = tf.subtract( dtrain, dx )
dist_sum = tf.reduce_sum( tf.square( dist ), 2 )
dist_min = tf.arg_min( dist_sum, 1 )        # 최근접 점의 인덱스를 반환

import random
shuffle_list = list( range( len( test ) ) ) # 299개
random.shuffle( shuffle_list )

import matplotlib.pyplot as plt
with tf.Session() as sess :
    init = tf.global_variables_initializer()
    sess.run( init )
    plt.plot( data[:,0], data[:,1], "ro", alpha=0.4 )
    
    correct = 0
    for i in range( 20 ) :
        index = shuffle_list[i]
        dist_sumout = sess.run( dist_sum, feed_dict={x:test_data[index]} )
        dist_minout = sess.run( dist_min, feed_dict={x:test_data[index]} )
        predict = train_label[ dist_minout ]
        y = test_label[index]        
        if np.argmax( y, 0 ) == np.argmax( predict, 1 ) :
            correct = correct + 1
        print( y, " : ", predict )
        plt.plot( test[index][0], test[index][1], alpha=0.5, color="k", marker="^")
print( "score : ", correct / 20 * 100 )           
plt.show()    