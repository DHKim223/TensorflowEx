# 행렬곱
import tensorflow as tf
x = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32, shape=[1,3])
y = tf.constant([2.0, 2.0, 2.0], dtype=tf.float32, shape=[3,1])
w = tf.matmul(x,y)
tf.print(w)

print(x.get_shape())        # (1,3)
print(y.get_shape())        # (3,1)
print(w.get_shape())        #(1,1)

a = tf.Variable([ [1,2], [3,4] ])
b = tf.Variable([ [5,6], [7,8] ])
tf.print(tf.add(a,b))
tf.print(tf.subtract(a,b))
tf.print(tf.multiply(a,b))
tf.print(tf.divide(a,b))

import numpy as np
print(np.multiply(a,b))
print(np.matmul(a,b))
print(np.sum(a))
print(np.mean(b))

tf.print(tf.reduce_sum(a))
tf.print(tf.reduce_mean(b))
tf.print(tf.argmax(a))      #    [1 1]
tf.print(tf.argmin(b))      #    [0 0]

# 브로드 캐스팅
c = tf.Variable([[1,2,3],[4,5,6] ] ,dtype=tf.float32)
d = tf.Variable( [[3],[3],[3]], dtype=tf.float32)

# 123      3
# 456      3
#             3
# tf.print(tf.matmul(c,d))
#
# e = tf.Variable([4],dtype=tf.float32)
# tf.print(tf.matmul(c,d)+e)
# tf.print(tf.matmul(ft.matmul(c,d)))       # stretch
#
# f =tf.Variable([[1,2,3],[4,5,6],[7,8,9],[11,12,13]])
# b = tf.Variable([3,3,3])
#
# tf.print(tf.add(f,g))
#
# h=tf.Variable

# reshape
c =np.arange(6)
print(c)
print(np.reshape(c,[2,3]))
print(np.reshape(c,[3,2]))
#print(np.reshape(c, [3,3]))         # 에러,,, 데이터가 9개가 있어야하기 때문에,,

d = tf.Variable([0,1,2,3,4,5])
tf.print(d)
tf.print(tf.reshape(d, [2,3]))
tf.print(tf.reshape(d, [3,2]))

