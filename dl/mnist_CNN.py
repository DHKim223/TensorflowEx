# # CNN
# # 손글씨 숫자 분석
# from tensorflow.keras.datasets import mnist
# ( train_data, train_label ), ( test_data, test_label ) = mnist.load_data()
# # print( train_data.shape )           # (60000, 28, 28)
# # print( train_label.shape )          # (60000,)
# # print( train_label )
#
# import matplotlib.pyplot as plt
# # plt.figure( figsize=( 8, 3 ) )
# # plt.subplots_adjust( wspace=0.5 )
# # plt.gray()
# # for i in range( 3 ) :
    # # plt.subplot( 1, 3, i+1 )
    # # img = train_data[i, :, :]       # 3차원
    # # plt.pcolor( 255 - img )
    # # plt.text( 24.5, 25, "%d" %(train_label[i]), color="blue", fontsize=15 )
    # # plt.xlim( 0, 27 )
    # # plt.ylim( 27, 0 )
    # # plt.grid( "on", color="white" )
# # plt.show()  
#
# # 1차원으로 바꿔서 분석
# from tensorflow.python.keras.utils import np_utils
# train_data = train_data.reshape(60000, 28*28)
# train_data = train_data.astype("float32")
# train_data = train_data / 255
# train_label = np_utils.to_categorical( train_label , 10)
#
# test_data = test_data.reshape(10000, 28*28)
# test_data = test_data.astype("float32")
# test_data = test_data / 255
# test_label = np_utils.to_categorical(test_label, 10)
#
# import numpy as np
# from tensorflow.keras import Sequential, layers
# np.random.seed(1)
# model = Sequential()
#
# model.add( layers.Dense( 64, input_dim=784, activation="relu" ))
# model.add( layers.Dense( 32, activation="relu"))
# model.add( layers.Dense( 10, activation="softmax"))
# model.compile( optimizer = "adam", loss ="categorical_crossentropy", \
               # metrics=["accuracy"])
# hist = model.fit( train_data, train_label, epochs = 10, batch_size = 1000, \
                  # validation_data = (test_data, test_label))
# score = model.evaluate( test_data, test_label)
# print( score )
#
# def show_prediction() :
    # n = 96
    # predicts = model.predict( test_data )
    # plt.figure( figsize=( 5, 5 ) )
    # plt.gray()
    # for i in range( n ) :
        # plt.subplot( 8, 12, i+1 )
        # data = test_data[i, :]
        # data = data.reshape( 28, 28 )
        # plt.pcolor( 1 - data )
        # predict = predicts[i, :]
        # label = np.argmax( predict )
        # plt.text( 22, 25.5, "%d" %label, fontsize=11 )
        # if label != np.argmax( test_label[i, :] ) :
            # plt.plot( [0,27], [1,1], color="blue", linewidth=5 )
        # plt.xlim( 0, 27 )
        # plt.ylim( 27, 0 )
        # plt.xticks( [], "" )
        # plt.yticks( [], "" )    
# # show_prediction()
# # plt.show()
#
# # 가중치
# # ws = model.layers[0].get_weights()[0]
# # plt.figure(figsize=(6,6))
# # plt.gray()
# # plt.subplots_adjust(wspace=0.3, hspace= 0.3)
# # for i in range( 64 ):
    # # plt.subplot(8, 8, i+1)
    # # w = ws[:,i]
    # # w = w.reshape(28, 28)
    # # plt.pcolor( 1 - w )
    # # plt.xlim( 0, 27)
    # # plt.ylim( 27, 0)
    # # plt.xticks( [], "")
    # # plt.yticks( [], "")
    # # plt.title( "%d" %i)
# # plt.show()
#
#

# # 2차원을 2차원으로 분석
# # 필터 추가
from tensorflow.keras.datasets import mnist
(train_data, train_label) , ( test_data, test_label) = mnist.load_data()
l, r, c = train_data.shape      # 60000, 28, 28
train_data = train_data.reshape( l, r, c, 1 )
train_data = train_data / 255.0
l, r, c = test_data.shape       # 10000, 28, 28
test_data = test_data.reshape( l, r, c, 1)
test_data = train_data / 255.0
print( train_data.shape )       # (60000, 28, 28, 1)
print( test_data.shape )        # (60000, 28, 28, 1)

import numpy as np
filter1 = np.array( [[ 1, 1, 1 ], [1, 1, 1] , [-2, -2, -2]], dtype=np.float32)
filter2 = np.array( [[-2, 1, 1], [-2, 1, 1], [-2, 1, 1]], dtype=np.float32)

index = 2
img = train_data[index, :, :]
img = img.reshape( r,c ) # 28 x 28
outimg1 = np.zeros_like( img )
outimg2 = np.zeros_like( img )

fr, fc = filter1.shape              # 3 x 3
for i in range( r - fr ):
    for j in range( c - fc ) :
        imgpart = img[i:i+3, j:j+3]
        outimg1[i+1, j+1] = np.dot( imgpart.reshape( -1), filter1.reshape(-1))
        outimg2[i+1, j+1] = np.dot( imgpart.reshape( -1), filter2.reshape(-1))
import matplotlib.pyplot as plt
plt.figure( figsize = ( 8,4) )
plt.gray()
plt.subplot(1, 3, 1)
plt.pcolor(1 - img )
plt.xlim( -1, 29 )
plt.ylim( 29 , -1)

plt.subplot(1,3,2)
plt.pcolor( 1 - outimg1 )
plt.xlim(-1, 29)
plt.ylim( 29, -1)

plt.show()

