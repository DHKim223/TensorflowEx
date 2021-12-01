# 타이타닉 생존자 예측

# pclass           1309 non-null int64         # 객실 등급
# survived         1309 non-null int64         # 생존 여부
# name             1309 non-null object        # 이름
# sex              1309 non-null object        # 성별
# age              1046 non-null float64       # 나이
# sibsp            1309 non-null int64         # 형제 자매 배우자 수
# parch            1309 non-null int64         # 부모 자식의 수. 0은 유모와 탑승
# ticket           1309 non-null object        # 티켓 번호
# fare             1308 non-null float64       # 요금
# cabin            295 non-null object         # 객실 번호
# embarked         1307 non-null object        # 정박항구. C=Cherbourg, Q=Queenstown, S=Southampton
# boat             486 non-null object         # 보트 탑승 여부
# body             121 non-null float64

import pandas as pd
dataset = pd.read_excel( "titanic.xls" )
# print( dataset.info() )
# print( dataset.describe() )

import matplotlib.pyplot as plt
import seaborn as sns
# 생존자 1 사망 0
# fig, ax = plt.subplots( 1, 2, figsize=( 8, 4) )
# dataset["survived"].value_counts().plot.pie( explode=[0, 0.1], \
        # autopct="%.2f%%", ax=ax[0] )
# ax[0].set_title( "survived" )
# sns.countplot( "survived", data=dataset, ax=ax[1] )
# ax[1].set_title( "survived" )

# 탑승자 연령
# dataset["age"].hist( bins=16, figsize=(5, 3), grid=False )
# plt.show()

# 객실 등급별 평균
# print( dataset.groupby( "pclass" ).mean() )

# 변수별 상관관계
# plt.figure( figsize=( 8, 4 ) )
# sns.heatmap( dataset.corr(), linewidths=0.01, \
            # square=True, annot=True, linecolor="white" )

# 연령별 객실등급별 성별과 생존률
# age_cut = pd.cut( dataset["age"], bins=[0, 10, 20, 50, 80], \
        # labels=["baby", "teenage", "adult", "old"] )
# plt.figure( figsize=( 8, 4 ) )
# plt.subplot( 131 )
# sns.barplot( age_cut, "survived", data=dataset )
# plt.subplot( 132 )
# sns.barplot( "pclass", "survived", data=dataset )
# plt.subplot( 133 )
# sns.barplot( "sex", "survived", data=dataset )

# 보트 탑승한 생존자 비율
# boat_survivors = dataset[ dataset["boat"].notnull() ]   # 미탑승자는 null
# fig, ax = plt.subplots( 1, 2, figsize=( 8, 4 ) )
# boat_survivors["survived"].value_counts().plot.pie( \
                # explode=[0, 0.1], autopct="%.2f%%", ax=ax[0] )
# ax[0].set_title( "boat_survived" )
# sns.countplot( "survived", data=boat_survivors, ax=ax[1] )
# ax[1].set_title( "survived" )
# plt.show()

# 성별 전처리
# print( dataset["sex"] )
import numpy as np
tmp = []
for data in dataset["sex"] :
    if data == "female" :
        tmp.append( 1 )
    elif data == "male" :
        tmp.append( 0 )
    else :
        tmp.append( np.nan )
dataset["sex"] = tmp        

# 결측값 제거
dataset = dataset[ dataset["age"].notnull() ]
dataset = dataset[ dataset["sibsp"].notnull() ]
dataset = dataset[ dataset["parch"].notnull() ]
dataset = dataset[ dataset["fare"].notnull() ]
# print( dataset.info() )

data = dataset.values[:, [0, 3, 4, 5, 6, 8]]
    # 객실등급 성별 나이 형제자매배우자수 부모자식의수 요금
label = dataset.values[:, [1]]      # 생존여부
from sklearn.model_selection import train_test_split
train_data, test_data, train_label, test_label = \
    train_test_split( data, label, test_size=0.3, random_state=0 )

train_data = np.asarray( train_data, np.float32 )
train_label = np.asarray( train_label, np.float32 )
test_data = np.asarray( test_data, np.float32 )
test_label = np.asarray( test_label, np.float32 )

# 모델 생성
from tensorflow.keras import Sequential, layers
model = Sequential()
model.add(layers.Dense(255, input_shape=(6, ) ,activation="relu"))
model.add( layers.Dense(1, activation="sigmoid"))
model.compile(optimizer="Adam", loss="mse", metrics=["accuracy"])
print(model.summary())

# 학습
hist = model.fit(train_data, train_label, validation_data=(test_data, test_label), epochs=500) 
# plt.figure(figsize=(8,4))
# plt.plot(hist.history["loss"])
# plt.plot(hist.history["val_loss"])
# plt.plot(hist.history["accuracy"])
# plt.plot(hist.history["val_accuracy"])
# plt.legend(["loss","val_loss","accuracy","val_accuracy"])
# plt.show()

# 예측
#person1 = np.array([3,0,19,0,0,5]).reshape(1,6)
#print(model.predict(person1))         #  [[0.14625728]]
# person2 = np.array([1,1,17,1,2,100]).reshape(1,6)
# print(model.predict(person2))   # [[0.99993813]]











    
    

























