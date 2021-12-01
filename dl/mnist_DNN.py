# ANN 인공신경망  Artificial Neural Network      신경망 구조를 모방한 기계학습 알고리즘
# DNN 심층신경망  Deep Neural Network            은닉층을 많이 늘려서 학습결과를 향상
# CNN 합성곱신경망 Convolution Neural Network    데이터의 특징을 추출해서 패턴을 분석
# RNN 순환신경망 Recurrent Neural Network        반복적이고 순차적인 학습에 특화. 시계열분석

# 심층신경망
from keras.utils import np_utils
from tensorflow.keras import layers, datasets, Sequential
(train_data, train_label), (test_data, test_label) = \
    datasets.mnist.load_data()
# print( train_data.shape )       # (60000, 28, 28)
# print( train_label.shape )      # (60000,)
    
# One Hot Encoding    
train_label = np_utils.to_categorical( train_label )
test_label = np_utils.to_categorical( test_label )
# print( train_label.shape )      # (60000, 10)
# print( test_label.shape )       # (10000, 10)

# 전처리 
l, w, h = train_data.shape
train_data = train_data.reshape( -1, w*h )
test_data = test_data.reshape( -1, w*h )

# 정규화
train_data = train_data / 255.0
test_data = test_data / 255.0
# print( train_data.shape )       # (60000, 784)
# print( test_data.shape )        # (10000, 784)

# 모델생성
model = Sequential()
model.add( layers.Dense( 100, activation="relu", \
                         input_shape=( 28*28, ), name="Hidden_1" ) )
model.add( layers.Dense( 50, activation="relu", name="Hidden_2" ) )
model.add( layers.Dense( 30, activation="relu", name="Hidden_3" ) )
model.add( layers.Dropout( 0.2 ) )
model.add( layers.Dense( 10, activation="softmax" ) )
model.compile( optimizer="adam", loss="categorical_crossentropy", \
               metrics=["accuracy"] ) 

# 모델 학습
hist = model.fit( train_data, train_label, validation_split=0.2, \
           batch_size=100, epochs=5 )

# 모델 평가
score = model.evaluate( test_data, test_label, batch_size=100 )
print( score )      # 손실률 정확도   은닉층 3 [0.10302143543958664, 0.9714999794960022]
                    #           은닉층 1 [0.10134143382310867, 0.9689000248908997]   

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax1 = ax.twinx()
ax.plot( hist.history["loss"], "y", label="train_loss" )
ax.set_ylabel( "loss" )
ax.legend( loc="upper left" )
ax1.plot( hist.history["accuracy"], "b", label="train_acc" )
ax1.set_ylabel( "acc" )
ax1.legend( loc="lower left" )
plt.show()

















