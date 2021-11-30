from tensorflow import keras
from tensorflow.keras import layers, datasets, Sequential
(train_data, train_label), (test_data, test_label) = \
    datasets.boston_housing.load_data()
# print( train_data.shape )           # (404, 13)
# print( train_label.shape )          # (404,)
# print( train_data )                
# print( train_label )          

# 데이터 정규화
mean = train_data.mean( axis=0 )
std = train_data.std( axis=0 )
train_data = ( train_data - mean ) / std 

mean = test_data.mean( axis=0 )
std = test_data.std( axis=0 )
test_data = ( test_data - mean ) / std

# 모델 생성
def bulid_model() :
    model = Sequential()
    model.add( layers.Dense( 64, activation="relu", \
                             input_shape=(train_data.shape[1],) ) )
    model.add( layers.Dense( 64, activation="relu" ) )
    model.add( layers.Dense( 1 ) )
    model.compile( optimizer="rmsprop", loss="mse", metrics=["mae"] )
    return model

model = bulid_model()
model.fit( train_data, train_label, epochs=100 )
predicts = model.predict( test_data )
# print( predicts.shape )
# print( test_label.shape )

import matplotlib.pyplot as plt
fig = plt.figure( figsize=( 8, 4 ) )
ax1 = fig.add_subplot( 1, 2, 1 )
ax1.scatter( test_label , predicts, alpha=0.5 )
ax1.set_xlabel( "label" )
ax1.set_ylabel( "predict" )

# K-겹 교차검증
import numpy as np
k = 4
samples = len(train_data) // k
scores = []
for i in range( k ) :
    # 검증용 데이터 1/4
    val_data = train_data[i*samples:(i+1)*samples]
    val_label = train_label[i*samples:(i+1)*samples]
    
    # 학습용데이터 3/4
    partial_data = np.concatenate( [train_data[:i*samples], train_data[(i+1)*samples:]], axis=0 )
    partial_label = np.concatenate( [train_label[:i*samples], train_label[(i+1)*samples:]], axis=0 )
    model = bulid_model()
    model.fit( partial_data, partial_label, epochs=100, batch_size=1 )
    mae = model.evaluate( val_data, val_label )
    scores.append( mae )
print( scores )
print( np.mean( scores ) )    

predicts = model.predict( test_data )
# print( predicts.shape )         # (102, 1)
# print( test_label.shape )       # (102,)
test_label = np.reshape( test_label, (102,1) )
result = np.concatenate( [predicts, test_label], axis=1 )
print( result )

ax2 = fig.add_subplot( 1, 2, 2 )
ax2.scatter( test_label, predicts, alpha=0.5)
ax1.set_xlabel( "label" )
ax1.set_ylabel( "predict" )
plt.show()









       
