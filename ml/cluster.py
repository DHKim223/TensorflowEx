# K 평균 알고리즘
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
filename = "label_dots.txt"
data = np.genfromtxt(filename, dtype=np.float32)
x = data[:, 0]
y = data[:, 1]
cluster = 3
count = len ( data )
#print( count )
centroid_x = []
centroid_y = []

indices = list( range(len(data)))
np.random.shuffle(indices)
#print(indices)
for i in range( cluster):
    centroid_x.append(x[ indices[i] ] )
    centroid_y.append(x[ indices[i] ] )
#print(centroid_x, centroid_y)

x_points = tf.constant(x)
y_points = tf.constant(y)

centroid_x_point = tf.placeholder(tf.float32, [cluster, ])      #( 899, )
centroid_y_point = tf.placeholder(tf.float32, [cluster, ])

print( x_points.shape ) #(899, )
reshape_x = tf.reshape( x_points, [1, -1])  # 1행 899열 2차원 배열
reshape_y = tf.reshape( y_points, [1, -1])
reshape_centroid_x = tf.reshape( centroid_x_point, [-1, 1] )    # 3행 1열
reshape_centroid_y = tf.reshape( centroid_y_point, [-1, 1] )    

sub_x = tf.subtract( reshape_x, reshape_centroid_x)
sub_y = tf.subtract( reshape_y, reshape_centroid_y)
distances = tf.square( sub_x ) + tf.square( sub_y )
assignment = tf.arg_min( distances, 0)                     #1행 899열 가까운 중심점 인덱스 

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for step in range(100):
    sorted_cluster_x = []
    sorted_cluster_y = []
    for i in range(cluster) :
        sorted_cluster_x.append([])
        sorted_cluster_y.append([])
    feed_dict = {centroid_x_point:centroid_x, centroid_y_point:centroid_y}
    assignmentout = sess.run( assignment, feed_dict = feed_dict)
    
    for i in range(count) :
        add_index = assignmentout[i]
        sorted_cluster_x[add_index].append(x[i])
        sorted_cluster_y[add_index].append(y[i])
    
    for i in range( cluster ):
        centroid_x[i] = np.mean(sorted_cluster_x[i])
        centroid_y[i] = np.mean(sorted_cluster_y[i])

import matplotlib.pyplot as plt
color = ["ro","go","bo"]
for i in range(cluster) :
    plt.plot(sorted_cluster_x[i], sorted_cluster_y[i], color[i], alpha=0.5)
    plt.plot(centroid_x[i], centroid_y[i], "k^")
plt.show()
        


