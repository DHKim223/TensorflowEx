#DNN
#https://www.tensorflow.org/tutorials/keras/classification?hl=ko
import tensorflow as tf
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_data, train_label), ( test_data, test_label) \
    = fashion_mnist.load_data()
print("다운로드 완료")
# print(train_data.shape)         #    (60000, 28, 28)
# print(train_label.shape)        #    (60000,)
# print(train_label)                   #    [9 0 0 ... 3 0 5]

# 레이블    클래스
#     0    T-shirt/top
#     1    Trouser
#     2    Pullover
#     3    Dress
#     4    Coat
#     5    Sandal
#     6    Shirt
#     7    Sneaker
#     8    Bag
#     9    Ankle boot

class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]

import matplotlib.pyplot as plt
# 이미지 확인
# plt.figure(figsize=(5,5))
# plt.imshow(train_data[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

train_data = train_data / 255.0
test_data = test_data / 255.0

# plt.figure( figsize = (8 , 8))
# for i in range( 25 ) :
    # plt.subplot(5, 5, i+1)
    # plt.xticks([])
    # plt.yticks([])
    # plt.grid(False)
    # plt.imshow( train_data[i], cmap=plt.cm.binary)
    # plt.xlabel( class_names[train_label[i]])
# plt.show()

model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28)),       # 1 차원배욜로 변환
        tf.keras.layers.Dense( 128, activation="relu" ),
        tf.keras.layers.Dense( 64, activation="relu"),
        tf.keras.layers.Dense( 10 )
    ])
model.compile(optimizer = "adam", \
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])
hist = model.fit(train_data, train_label, epochs=10)
loss, acc = model.evaluate(test_data, test_label, verbose=2)
print(loss, acc)    # test # 0.8815000057220459
loss, acc = model.evaluate(train_data, train_label, verbose=2)
print(loss, acc)    # train # 0.9122499823570251

# 예측
import numpy as np
model = tf.keras.Sequential( [model, tf.keras.layers.Softmax()])
predicts = model.predict(test_data)
print(predicts[0])
print(np.argmax(predicts[0]))
print(test_label[0])

def plot_image(i, predict, label, img):
    label, img = label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predict_label = np.argmax(predict)
    if predict_label == label :
        color = "blue"
    else :
        color = "red"
    plt.xlabel("{} {:2.0f}% ({})".format( class_names[predict_label], 
                                          100*np.max(predict), class_names[label], color=color))

def plot_value_array(i, predict, label):
    label = label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar( range(10), predict, color="#777777")
    plt.ylim([0,1])
    predict_label = np.argmax(predict)
    thisplot[predict_label].set_color("red")
    thisplot[label].set_color("blue")
    
# index = 12    
# plt.figure( figsize = (8,4))
# plt.subplot(1,2,1)
# plot_image(index,predicts[0],test_label, test_data)
# plt.subplot(1,2,2)
# plot_value_array(index, predicts[0],test_label)
# plt.show()

# rows = 5
# cols = 3
# number = rows * cols
# plt.figure(figsize=(8,8))
# for i in range (number): 
    # plt.subplot( rows, cols*2, i*2 +1)
    # plot_image(i, predicts[i],test_label, test_data)
    # plt.subplot(rows,cols*2, i*2+2)
    # plot_value_array(i, predicts[i],test_label)
# plt.tight_layout()
# plt.show()

img = test_data[1000]
#print(img.shape)        (28, 28)
img = (np.expand_dims(img,0))
#print(img.shape)        (1,28,28)

predict = model.predict(img)
print(predict)

plot_value_array(1000,predict[0],test_label)
plt.xticks( range(10), class_names, rotation=45)
plt.show()
print(np.argmax( predict[0]))
