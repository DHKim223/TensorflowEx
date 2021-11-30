# import tensorflow as tf
# print(tf.__version__)
#
# hello = tf.constant("Hello Tensorflow!")
# tf.print(hello)

# 1버전        
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# a = tf.constant([5],dtype=tf.float32)
# b = tf.constant([10],dtype=tf.float32)
# c = tf.constant([2],dtype=tf.float32)
# d = a*b+c
# #tf.print(d)
# sess = tf.Session()
# result = sess.run(d)
# print(result)
#
# # 플레이스 홀더
# input_data = [1,2,3,4,5]
# x = tf.placeholder(dtype=tf.float32)
# y = x * 2
# sess = tf.Session()
# result = sess.run(y, feed_dict={x:input_data})
# print(result)               # tensorflow 계산
# print(type(result))
#
# result = input_data * 2
# print(result)           # python 계산,,, 12345 가 두번 출력된다.

# 변수
# 1버전
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# input_data = [1,2,3,4,5]
# x = tf.placeholder(tf.float32)
# b = tf.Variable([10], dtype=tf.float32)
# W = tf.Variable([0.5],dtype=tf.float32)
# y = W * x + b
# session = tf.Session()
# init = tf.global_variables_initializer()
# session.run(init)
#
# result = session.run(y, feed_dict={x:input_data})
# print(result)

# 2버전
# import tensorflow as tf
# score = [56, 34, 54, 28, 86]
# a = tf.Variable([2], dtype=tf.float32)      # tf 변수
# print(score * a)
# tf.print(score*a)
#
# a= 10       # 파이썬 변수
# print(score*a)  # score 10번 출력
# tf.print(score * a ) # 

# 함수
# import tensorflow as tf
# a = tf.compat.v1.Variable( [5], dtype = tf.float32)
# b = tf.compat.v1.Variable( [2], dtype = tf.float32)
# tf.print(tf.add(a,b))
# tf.print(tf.subtract(a,b))
# tf.print(tf.multiply(a,b))
# tf.print(tf.divide(a,b))
#
# c = tf.compat.v2.Variable( [5], dtype = tf.float32)
# d = tf.compat.v2.Variable( [2], dtype = tf.float32)
# tf.print(tf.add(c,d))
# tf.print(tf.subtract(c,d))
# tf.print(tf.multiply(c,d))
# tf.print(tf.divide(c,d))

import tensorflow as tf
c = tf.Variable( [-5], dtype = tf.float32)
d = tf.Variable( [2], dtype = tf.float32)
#tf.print(tf.mod(c,d))           # 에러발생 
tf.print(tf.compat.v1.mod(c,d))
tf.print(tf.abs(c))
tf.print(tf.square(c))
tf.print(tf.sqrt(d))
tf.print(tf.maximum(c,d))
tf.print(tf.minimum(c,d))


