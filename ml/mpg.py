# 자동차 연비 예측
# https://www.tensorflow.org/tutorials/keras/regression?h1=ko
from tensorflow import keras
path = keras.utils.get_file("auto-mpg.data", \
                                        "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
# print("다운로드 완료")

import pandas as pd
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
dataset = pd.read_csv(path, names = column_names, na_values="?", \
                      comment="\t", sep=" ", skipinitialspace=True)
dataset = dataset.copy()
# print(type(dataset))        #데이터 타입확인
# print( dataset.tail() )         # 데이터 확인
# print(dataset.shape)      # (398, 8)
# print(dataset.isna().sum())     # horsepower에 na 6개
dataset = dataset.dropna()      # na 드랍
# print( dataset.shape)       # (392, 8 )

# One Hot Encoding
origin = dataset.pop("Origin")
# print(origin.shape)                     #(392, )
dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
# print(dataset.tail())

# 데이터셋을 훈련 세트와 테스트 세트로 분할
train = dataset.sample(frac=0.8, random_state=0)
test = dataset.drop(train.index)

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns
# sns.pairplot(train[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
# plt.show()

# 통계화
train_stats = train.describe()
#print(train_stats)
train_stats.pop("MPG")
train_stats = train_stats.transpose()
#print(train_stats)

# 특성과 레이블 분리
train_label = train.pop("MPG")
test_label = test.pop("MPG")

# 데이터 정규화
train_data = (train - train_stats["mean"])/ train_stats["std"]
test_data = (test - train_stats["mean"])/train_stats["std"]
#print(train_data.head())

# 모델 만들기

model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=[len(train_data.keys())]),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1) ])
optimizer = keras.optimizers.RMSprop(0.001)
model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])

# hist = model.fit (train_data, train_label, epochs=100, validation_split=0.2)
# #print(hist.history)
#
# history = pd.DataFrame(hist.history)
# history ["epoch"] = hist.epoch
# print(history.tail())

#epoch 100번 돌려도 loss가 작아지지 않으니,,, 적당한 시점에 끊기
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss",patience=10)
hist = model.fit(train_data,train_label, epochs=100, validation_split=0.2, callbacks=[early_stop])
print(hist.history)
history = pd.DataFrame(hist.history)
history["epoch"] = hist.epoch

# plt.figure(figsize=(8,5))
# plt.subplot( 1,2,1)
# plt.xlabel("Epoch")
# plt.ylabel("MAE")
# plt.plot(history["epoch"], history["mae"], label="Train Error")
# plt.plot(history["epoch"], history["val_mae"], label="Val Error")
# plt.ylim([0,5])
# plt.legend()
# plt.show()

loss, mae, mse = model.evaluate( test_data, test_label )
print( mae )

predicts = model.predict( test_data ).flatten()
# plt.scatter( test_label, predicts )
# plt.xlabel( "True Values" )
# plt.ylabel( "Predictions" )
# plt.xlim( [0, plt.xlim()[1] ] )
# plt.ylim( [0, plt.xlim()[1] ] )
# _ = plt.plot( [-100, 100], [-100, 100] )
# plt.show()

error = predicts - test_label
plt.hist( error, bins=25 )
plt.xlabel( "Prediction Error" )
plt.ylabel( "Count" )
plt.show() 


