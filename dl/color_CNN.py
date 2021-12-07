# 컬러 이미지 분석
# 비행기 자동차 새 고양이 사슴 개 개구리 말 배 트럭
from tensorflow.keras import datasets
( train_data, train_label), (test_data, test_label ) \
    = datasets.cifar10.load_data()
# print( train_data.shape )            # (50000, 32, 32, 3)
# print( train_label.shape )           # (50000, 1)
# print( train_label )

from tensorflow.python.keras.utils import np_utils
train_label = np_utils.to_categorical( train_label )
test_label = np_utils.to_categorical( test_label )
# print( train_label )
train_data = train_data.astype( "float32" )
test_data = test_data.astype( "float32" )
train_data /= 255
test_data /= 255

from tensorflow.keras import Sequential, layers
model = Sequential()
model.add( layers.Conv2D( 64, (3, 3), padding="same", activation="relu", \
                          input_shape=( 32, 32, 3 ) ) )
model.add( layers.Conv2D( 64, (3, 3), activation="relu" ) )
model.add( layers.MaxPooling2D( pool_size=(2, 2) ) )
model.add( layers.Dropout( 0.25 ) )

model.add( layers.Conv2D( 256, (3, 3), padding="same", activation="relu" ) )
model.add( layers.Conv2D( 256, (3, 3), activation="relu" ) )
model.add( layers.MaxPooling2D( pool_size=(2, 2) ) )
model.add( layers.Dropout( 0.25 ) )

model.add( layers.Flatten() )
model.add( layers.Dense( 10, activation="relu" ) )
model.add( layers.Dropout( 0.25 ) )
model.add( layers.Dense( 10, activation="softmax" ) )
model.compile( loss="categorical_crossentropy", optimizer="adam", \
               metrics=["accuracy"] )
# print( model.summary() )

hist = model.fit( train_data, train_label, epochs=3, batch_size=32, \
           validation_data=( test_data, test_label ) )
score = model.evaluate( test_data, test_label, batch_size=32 )
print( score ) 


















