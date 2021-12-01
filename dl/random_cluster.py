# DNN
# 난수 데이터 군집화
import numpy as np
np.random.seed(1)
n = 200                 # 데이터 수
k = 3                   # 군집
label = np.zeros( (n, 3), dtype=np.uint8 )  # 정답
train = np.zeros( (n, 2) )                  # 데이터
centers = np.array( [[-.5, -.5], [.5, 1.], [1., -.5]] )   # 중심점
std = np.array( [[.5, .5], [.5, .3], [.3, .5]] )          # 분산
pi = np.array([0.4, 0.8, 1])    # 0.4 보다 작은 경우 0 0.8보다 작은 경우 1 그 외는 2
for i in range( n ) :
    wk = np.random.rand()
    for j in range( k ) :
        if wk < pi[j] :
            label[i, j] = 1
            break
    for j in range( 2 ) :
        train[i, j] = np.random.randn() * std[ label[i,:] == 1, j ] \
                    + centers[ label[i, :] == 1, j] 

training = int( n * 0.7 )
train_data = train[:training, :]
train_label = label[:training, :]
test_data = train[training:, :]
test_label = label[training:, :]

np.savez( "class_data.npz", train_data=train_data, train_label=train_label, \
          test_data=test_data, test_label=test_label )

import matplotlib.pyplot as plt
# colors = ["red", "green", "blue"]
# print( train_label.shape )               # (140, 3)
# n, c = train_label.shape
# for i in range( c ) :
    # plt.plot( train_data[ train_label[:, i]==1, 0], \
              # train_data[ train_label[:, i]==1, 1], \
              # linestyle="none", marker="o", color=colors[i], alpha=0.5 )
# plt.show()  

# 데이터 로드
dataset = np.load( "class_data.npz" )
train_data = dataset["train_data"]
train_label = dataset["train_label"]
test_data = dataset["test_data"]
test_label = dataset["test_label"]
# print( train_data.shape )           # (210, 2)
# print( train_label.shape )          # (210, 3)

# 모델 생성
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import SGD
model = Sequential()
model.add( layers.Dense( 2, input_dim=2, activation="sigmoid", \
                         kernel_initializer="uniform" ) )
model.add( layers.Dense( 3, activation="softmax", kernel_initializer="uniform" ) )
sgd = SGD( lr=1, momentum=0.0, decay=0.0, nesterov=False )
model.compile( optimizer=sgd, loss="categorical_crossentropy", \
               metrics=["accuracy"] )

# 학습
hist = model.fit( train_data, train_label, epochs=50, batch_size=10, \
                  validation_data=( train_data, train_label ) )
score = model.evaluate( test_data, test_label )
print( score )

plt.figure( figsize=( 8, 4 ) )
plt.subplots_adjust( wspace=0.5 )
plt.subplot( 1, 3, 1 )
plt.plot( hist.history["loss"], "b", label="training" )
plt.plot( hist.history["val_loss"], "cornflowerblue", label="test" )
plt.legend()

plt.subplot( 1, 3, 2 )
plt.plot( hist.history["accuracy"], "r", label="training" )
plt.plot( hist.history["val_accuracy"], "y", label="test" )
plt.legend()

plt.subplot( 1, 3, 3 )
colors = ["red", "green", "blue"]
n, c = train_label.shape
for i in range( c ) :
    plt.plot( test_data[ test_label[:, i]==1, 0],\
              test_data[ test_label[:, i]==1, 1],\
              linestyle="none", marker=".", color=colors[i], alpha=0.5 )

xn = 60
a = np.linspace( -3, 3, xn )
b = np.linspace( -3, 3, xn )
aa, bb = np.meshgrid( a, b )
# print( aa.shape )               # (60, 60)        
xx = np.c_[ np.reshape( aa, xn*xn, order="C"), \
           np.reshape( bb, xn*xn, order="C")]
# print( xx.shape )               # (3600, 2)
yy = model.predict( xx )
# print( yy.shape )               # (3600, 3)

for i in range( k ) :
    f = yy[:, i]
    f = f.reshape( xn, xn )
    f = f.T
    cont = plt.contour( aa, bb, f, levels=[0.5, 0.9], \
                        colors=["gray", "black"])
    cont.clabel( fmt="%.1f", fontsize=7 )
    plt.xlim( -3, 3 )
    plt.ylim( -3, 3 )

plt.show()
