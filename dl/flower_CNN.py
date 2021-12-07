# 컬러 꽃 이미지 분석
# tensorflow.org/tutorials/images/classification?hl=ko

from tensorflow import keras 
import pathlib
url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
dir = keras.utils.get_file( "flower_photos", origin=url, untar=True )
dir = pathlib.Path( dir )
count = len( list( dir.glob( "*/*.jpg" ) ) )
print( count )              # 3670

train = keras.preprocessing.image_dataset_from_directory(
    dir, validation_split=0.2, subset="training", seed=123, \
    image_size=( 180, 180 ), batch_size=32 )
test = keras.preprocessing.image_dataset_from_directory( 
    dir, validation_split=0.2, subset="validation", seed=123,\
    image_size=( 180, 180 ), batch_size=32 )

class_names = train.class_names
# print( class_names )        #['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

import matplotlib.pyplot as plt
# 데이터 확인 ( 사진확인 )
# plt.figure(figsize= (8,8))
# for image, label in train.take(2):
    # for i in range(9):
        # ax = plt.subplot(3,3,i+1)
        # plt.imshow( image[i].numpy().astype("uint8"))
        # plt.title(class_names[label[i] ] )
        # plt.axis("off")
# plt.show()

for image_batch, label_batch in train :
    print( image_batch.shape)   # (32, 180, 180, 3)
    print( label_batch.shape)       #(32, )
    break

import tensorflow as tf
autotune = tf.data.experimental.AUTOTUNE
train_data = train.cache().shuffle(1000).prefetch(buffer_size=autotune)
test=test.cache().prefetch(buffer_size=autotune)

from tensorflow.keras import layers
normalization = layers.experimental.preprocessing.Rescaling(1./255)

import numpy as np
train_normal = train.map( lambda x, y : ( normalization(x), y) )
image_batch, label_batch = next( iter( train_normal ) )
first_image = image_batch[0]
print( np.min( first_image), np.max( first_image ) )    # 0.0 1.0

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, Dropout

# 데이터 증강
augmentation = Sequential([
    layers.experimental.preprocessing.RandomFlip( "horizontal",\
            input_shape=( 180, 180, 3)),
    layers.experimental.preprocessing.RandomRotation( 0.1 ),
    layers.experimental.preprocessing.RandomZoom( 0.1 )
    ])

plt.figure( figsize=( 6, 6 ) )
for img, _ in train.take( 1 ) :
    for i in range( 9 ) :
        images = augmentation( img )
        plt.subplot( 3, 3, i+1 )
        plt.imshow( images[0].numpy().astype( "uint8" ) )
        plt.axis( "off" )
plt.show()        






model = Sequential([
        layers.experimental.preprocessing.Rescaling( 1./255, \
                    input_shape=(180, 180, 3) ),
        Conv2D( 16, 3, padding="same", activation="relu" ),
        MaxPooling2D(),    
        Dropout(0.25),
        Conv2D( 32, 3, padding="same", activation="relu" ),
        MaxPooling2D(),
        Dropout(0.25),
        Conv2D( 64, 3, padding="same", activation="relu" ),
        MaxPooling2D(),
        Dropout(0.25),
        Flatten(),
        Dense( 128, activation="relu" ),
        Dense( len( class_names ) )        
    ])
    
model.compile( optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), \
               metrics=["accuracy"] )
# print( model.summary() )
hist = model.fit( train, validation_data=test, epochs=10 )
#print( hist.history["loss"] )               # [1.3525466918945312, 1.0502370595932007, 0.8684704303741455, 0.6902374029159546, 0.49081286787986755, 0.3159932792186737, 0.20337918400764465, 0.1094731017947197, 0.10415273904800415, 0.04621339961886406]
#print( hist.history["accuracy"] )       # [0.4172343313694, 0.587193489074707, 0.669618546962738, 0.7394413948059082, 0.8215258717536926, 0.8896457552909851, 0.9298365116119385, 0.9659400582313538, 0.9666212797164917, 0.9884195923805237]
acc = hist.history["accuracy"]
val_acc = hist.history["val_accuracy"]
loss = hist.history["loss"]
val_loss = hist.history["val_loss"]

#
# plt.figure( figsize=(6,6))
# plt.subplot(1,2,1)
# plt.plot( range(10), acc, label="Training")
# plt.plot( range(10), val_acc, label="Validation")
# plt.legend(loc="lower right")
#
# plt.subplot(1,2,2)
# plt.plot( range(10), loss, label="Training")
# plt.plot( range(10), val_loss, label="Validation")
# plt.show()

url = "https://storage.googleapis.com/download.tensorflow.org/example_images/592px-Red_sunflower.jpg"
path = keras.utils.get_file( "red_sunflower", origin=url )
img = keras.preprocessing.image.load_img( path, target_size=( 180, 180) )
img_array = keras.preprocessing.image.img_to_array( img )
img_array = tf.expand_dims( img_array, 0)
predict = model.predict( img_array )
score = tf.nn.softmax( predict[0] )
print( class_names[ np.argmax( score) ], 100*np.max( score ) )  # sunflowers 99.99
