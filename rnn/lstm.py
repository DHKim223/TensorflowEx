# GRU            # numpy 1.19.5 버전으로 
# from tensorflow.keras import models, layers
# model = models.Sequential()
# model.add( layers.Embedding( input_dim=1000, output_dim=64 ) )
# model.add( layers.GRU( 256, return_sequences=True ) )
    # # batch_size, timesteps, 256 
# model.add( layers.SimpleRNN( 128 ) )
    # # batch_size, 128
# model.add( layers.Dense( 10 ) )
# print( model.summary() ) 
    # # batch_size, timesteps, units
    
    
# LSTM
# 인코더-디코더 모델
# from tensorflow.keras import models, layers
# encoder = 1000
# decoder = 2000
# encoder_input = layers.Input( shape=(None,) )
# encoder_embeded = layers.Embedding( input_dim=encoder, \
                            # output_dim=64 )( encoder_input )
# output, state_h, state_c = \
    # layers.LSTM( 64, return_state=True, name="encoder")(encoder_embeded)
    #
# encoder_state = [ state_h, state_c ]
# decoder_input = layers.Input( shape=(None, ) )
# decoder_embeded = layers.Embedding( input_dim=decoder, \
                            # output_dim=64 )( decoder_input )
# decoder_output = layers.LSTM( 64, name="decoder" )\
                # ( decoder_embeded, initial_state=encoder_state )
# output = layers.Dense( 10 )( decoder_output )
# model = models.Model( [encoder_input, decoder_input], output )
# print( model.summary() )                                    


# 양방향 RNN
# from tensorflow.keras import models, layers
# model = models.Sequential()
# model.add( layers.Bidirectional(\
         # layers.LSTM( 64, return_sequences=True), input_shape=( 5, 10 )))
# model.add( layers.Bidirectional( layers.LSTM( 32 ) ) )
# model.add( layers.Dense( 10 ) )
# print( model.summary())


# 병렬처리 모델
# from tensorflow.keras import models, layers
# batch_size = 64
# input_dim = 28
# units = 64
# output = 10
# def build_model( cudnn=True ) :
    # if cudnn :
        # lstm_layer = layers.LSTM( units, input_shape=(None, input_dim ) )
    # else :
        # lstm_layer = layers.RNN( layers.LSTMCell(units), \
                            # input_shape=( None, input_dim ) )
    # model = models.Sequential( \
            # [lstm_layer, layers.BatchNormalization(), layers.Dense( output )] )    
    # return model
# from tensorflow.keras.datasets import mnist
# (train_data, train_label), (test_data, test_label) =\
    # mnist.load_data()
    #
# from tensorflow.keras.losses import SparseCategoricalCrossentropy
# import time
#
# # 병렬처리 안 하는 경우
# model = build_model( cudnn=True )      # 47.536951780319214    # 46.613293409347534
# model.compile( optimizer="sgd", loss=SparseCategoricalCrossentropy(from_logits=True), \
               # metrics=["accuracy"] )
# start = time.time()
# model.fit( train_data, train_label, epochs=5, \
           # validation_split=0.2, batch_size=batch_size )
# end = time.time()
# print( end - start )    


# 날짜별로 특정한 값이 있는 데이터
import pandas as pd
from pyspark.sql.functions import column
df = pd.read_csv( "cansim-0800020-eng-6674700030567901031.csv", \
             skiprows=6, skipfooter=9, engine="python" )
# print( df.head() )

from pandas.tseries.offsets import MonthEnd
df["Adjustments"] = pd.to_datetime( df["Adjustments"] ) + MonthEnd( 1 )
# print( df.head() )

import matplotlib.pyplot as plt
# df.plot()

split_date = str( pd.Timestamp( "01-01-2011" ) )
train = df.loc[ :split_date, ["Unadjusted"] ]
test = df.loc[split_date:, ["Unadjusted"] ]
ax = train.plot()
# test.plot( ax = ax)
# plt.legend(["train", "test"])
# plt.show()

# 정규화
from sklearn.preprocessing import MinMaxScaler
mm = MinMaxScaler()
train_data = mm.fit_transform( train )
test_data = mm.fit_transform( test )
# print( train_data.shape )           # (202, 1)
# print( test_data.shape )            # (111, 1)
# print( train_data )

train_df = pd.DataFrame( train_data, columns=["scaled"], \
                        index=train.index )
test_df = pd.DataFrame( test_data, columns=["scaled"], \
                        index=test.index )
for s in range( 1, 13 ) :
    train_df["shift{}".format(s)] = train_df["scaled"].shift( s )
    test_df["shift{}".format(s)] = test_df["scaled"].shift( s )
# print( train_df.head( 15 ) )    

train_data = train_df.dropna()
test_data = test_df.dropna()
train_label = train_data[ "scaled" ]
test_label = test_data[ "scaled" ]
train_data = train_data.drop( "scaled", axis=1 )
test_data = test_data.drop( "scaled", axis=1 )
# print( train_data.values.shape )            # (190, 12)
# print( train_label.values.shape )           # (190,)
# print( test_data.values.shape )             # (99, 12)
# print( test_label.values.shape )            # (99,)

#print(test_data)
print(test_label)
train_data = train_data.values
test_data = test_data.values
train_label = train_label.values
test_label = test_label.values

train_data = train_data.reshape(train_data.shape[0], 12, 1)
test_data = test_data.reshape( test_data.shape[0], 12, 1)
# print(train_data.shape)  #(190, 12, 1)
# print(test_data.shape)  # (99, 12, 1)

import tensorflow.keras.backend as K
from tensorflow.keras import models, layers
K.clear_session()
model = models.Sequential()
model.add( layers.LSTM( 20, input_shape=(12,1)))
model.add( layers.Dense(1))
model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
# print(model.summary())
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                Output Shape              Param #   
# =================================================================
# lstm (LSTM)                 (None, 20)                1760      
#
# dense (Dense)               (None, 1)                 21        
#
# =================================================================
# Total params: 1,781
# Trainable params: 1,781
# Non-trainable params: 0
# _________________________________________________________________
# None

model.fit(train_data, train_label, epochs=100, batch_size=30)
predicts = model.predict(test_data)
# for i in range (len(predicts )):
    # print(predicts[i], "\t", test_label[i])

predict_df = pd.DataFrame(predicts)
label_df = pd.DataFrame(test_label)
ax = label_df.plot()
predict_df.plot(ax=ax)
plt.legend(["label","predict"])
plt.show()


# import numpy as np
# import tensorflow_datasets as tfds
# import tensorflow as tf
#
# tfds.disable_progress_bar()
#
# import matplotlib.pyplot as plt
# def plot_graphs(history, metric):
    # plt.plot(history.history[metric])
    # plt.plot(history.history['val_'+metric], '')
    # plt.xlabel("Epochs")
    # plt.ylabel(metric)
    # plt.legend([metric, 'val_'+metric])
    #
# dataset, info = tfds.load('imdb_reviews', with_info=True,
                          # as_supervised=True)
# train_dataset, test_dataset = dataset['train'], dataset['test']
# print( train_dataset.element_spec )
#
# for example, label in train_dataset.take(1):
    # print('text: ', example.numpy())
    # print('label: ', label.numpy())
    #
    #
# BUFFER_SIZE = 10000
# BATCH_SIZE = 64
#
# train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
# test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
#
# for example, label in train_dataset.take(1):
    # print('texts: ', example.numpy()[:3])
    # print()
    # print('labels: ', label.numpy()[:3])    
    #
    #
# VOCAB_SIZE = 1000
# encoder = tf.keras.layers.experimental.preprocessing.TextVectorization(
    # max_tokens=VOCAB_SIZE)
# encoder.adapt(train_dataset.map(lambda text, label: text))
#
# vocab = np.array(encoder.get_vocabulary())
# vocab[:20]    
    #
# encoded_example = encoder(example)[:3].numpy()
# print( encoded_example )    
    #
# for n in range(3):
    # print("Original: ", example[n].numpy())
    # print("Round-trip: ", " ".join(vocab[encoded_example[n]]))
    # print()    
    #
    #
# model = tf.keras.Sequential([
    # encoder,
    # tf.keras.layers.Embedding(
        # input_dim=len(encoder.get_vocabulary()),
        # output_dim=64,
        # # Use masking to handle the variable sequence lengths
        # mask_zero=True),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    # tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dense(1)
# ])    
# print([layer.supports_masking for layer in model.layers])
    #
# # predict on a sample text without padding.
#
# sample_text = ('The movie was cool. The animation and the graphics '
               # 'were out of this world. I would recommend this movie.')
# predictions = model.predict(np.array([sample_text]))
# print(predictions[0])    
    #
    #
# # predict on a sample text with padding
#
# padding = "the " * 2000
# predictions = model.predict(np.array([sample_text, padding]))
# print(predictions[0])
#
# model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              # optimizer=tf.keras.optimizers.Adam(1e-4),
              # metrics=['accuracy'])
              #
# history = model.fit(train_dataset, epochs=10,
                    # validation_data=test_dataset,
                    # validation_steps=30)
                    #
# test_loss, test_acc = model.evaluate(test_dataset)
#
# print('Test Loss:', test_loss)
# print('Test Accuracy:', test_acc)
#
#
# plt.figure(figsize=(16, 8))
# plt.subplot(1, 2, 1)
# plot_graphs(history, 'accuracy')
# plt.ylim(None, 1)
# plt.subplot(1, 2, 2)
# plot_graphs(history, 'loss')
# plt.ylim(0, None)
#
#
# sample_text = ('The movie was cool. The animation and the graphics '
               # 'were out of this world. I would recommend this movie.')
# predictions = model.predict(np.array([sample_text]))
#
#
# model = tf.keras.Sequential([
    # encoder,
    # tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    # tf.keras.layers.Dense(64, activation='relu'),
    # tf.keras.layers.Dropout(0.5),
    # tf.keras.layers.Dense(1)
# ])
#
# model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              # optimizer=tf.keras.optimizers.Adam(1e-4),
              # metrics=['accuracy'])
              #
# history = model.fit(train_dataset, epochs=10,
                    # validation_data=test_dataset,
                    # validation_steps=30)
                    #
# test_loss, test_acc = model.evaluate(test_dataset)
#
# print('Test Loss:', test_loss)
# print('Test Accuracy:', test_acc)
#
# # predict on a sample text without padding.
#
# sample_text = ('The movie was not good. The animation and the graphics '
               # 'were terrible. I would not recommend this movie.')
# predictions = model.predict(np.array([sample_text]))
# print(predictions)
#
# plt.figure(figsize=(16, 6))
# plt.subplot(1, 2, 1)
# plot_graphs(history, 'accuracy')
# plt.subplot(1, 2, 2)
# plot_graphs(history, 'loss')






            
            
            
            
            
            