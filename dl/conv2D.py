from tensorflow.keras import datasets
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras import Sequential, layers

(train_data, train_label), (test_data,test_label) = \
    datasets.mnist.load_data()
l, w, h = train_data.shape
train_data = train_data.reshape(l, w, h, 1).astype("float32")
train_data = train_data / 255
l, w, h = test_data.shape
test_data = test_data.reshape(l ,w, h, 1).astype("float32")
test_data = test_data / 255
train_label = np_utils.to_categorical(train_label, 10)
test_label = np_utils.to_categorical(test_label, 10)

model = Sequential()
model.add(layers.Conv2D(16, (3, 3), padding="same", input_shape=(28, 28, 1), activation="relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Conv2D(32,(3,3), activation="relu"))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add( layers.Dropout(0.25))
model.add(layers.Flatten())
    # 출력 4 차원 (배치수, 필터수, 이미지 세로폭, 이미지 가로폭)
    # Flatten ( 배치수 ,필터수 * 이미지 세로폭 * 이미지 가로폭 )
model.add( layers.Dense(128, activation="relu"))
model.add( layers.Dropout(0.25))
model.add( layers.Dense( 10, activation="softmax"))
model.compile( loss = "categorical_crossentropy", optimizer="adam", \
               metrics=["accuracy"])
model.fit( train_data, train_label, batch_size=1000, epochs=10, \
           validation_data=(test_data, test_label))
score = model.evaluate(train_data, train_label)
print(score)    # [loss , accuracy ] = [0.04624331369996071, 0.987766683101654]
score = model.evaluate(test_data, test_label)
print(score)

import matplotlib.pyplot as plt
import numpy as np
def show_prediction() :
    n = 96
    predicts = model.predict( test_data )
    plt.figure( figsize=( 5, 5 ) )
    plt.gray()
    for i in range( n ) :
        plt.subplot( 8, 12, i+1 )
        data = test_data[i, :]
        data = data.reshape( 28, 28 )
        plt.pcolor( 1 - data )
        predict = predicts[i, :]
        label = np.argmax( predict )
        plt.text( 22, 25.5, "%d" %label, fontsize=11 )
        if label != np.argmax( test_label[i, :] ) :
            plt.plot( [0,27], [1,1], color="blue", linewidth=5 )
        plt.xlim( 0, 27 )
        plt.ylim( 27, 0 )
        plt.xticks( [], "" )
        plt.yticks( [], "" )    
# show_prediction()
# plt.show()

plt.figure( figsize=( 8, 4 ) )
plt.gray()
plt.subplots_adjust( wspace=0.2, hspace=0.2 )
plt.subplot( 2, 9, 10 )
index = 12
img = test_data[index, :, :, 0]
img = img.reshape( 28, 28 )
plt.pcolor( 1 - img )
plt.xlim( 0, 28 )
plt.ylim( 28, 0 )
plt.xticks( [], "" )
plt.yticks( [], "" )
plt.title( "Original" )

w = model.layers[0].get_weights()[0]
max_w = np.max( w )
min_w = np.min( w )
# for i in range( 8 ) :
    # plt.subplot( 2, 9, i+2 )
    # w1 = w[:, :, 0, i]
    # w1 = w1.reshape( 3, 3 )
    # plt.pcolor( -w1, vmin=min_w, vmax=max_w )
    # plt.xlim( 0, 3 )
    # plt.ylim( 3, 0 )
    # plt.xticks( [], "" )
    # plt.yticks( [], "" )
    # plt.title( "%d" %i )
    #
    # plt.subplot( 2, 9, i+11 )
    # out_img = np.zeros_like( img )
    #
    # for j in range( 28 - 3 ) :
        # for k in range( 28 - 3 ) :
            # img_part = img[j: j+3, k:k+3]
            # out_img[ j+1, k+1] = \
                # np.dot( img_part.reshape( -1 ), w1.reshape( -1 ) )
    # plt.pcolor( -out_img )
    # plt.xlim( 0, 28 )
    # plt.ylim( 28, 0 )
    # plt.xticks( [], "" )
    # plt.yticks( [], "" )
    #
# plt.show()