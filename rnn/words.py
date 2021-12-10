# # 문장 만들기
# import codecs
# with codecs.open( "../word/토지1.txt", "r", encoding="ms949" ) as f :
    # text = f.read()
# # print( text )    
# # print( type( text ) )           # <class 'str'>
# chars = sorted( list( set( text ) ) )
# # print( chars )
# char_indexs = dict( ( c, i ) for i, c in enumerate( chars ) )
# indexs_char = dict( ( i, c ) for i, c in enumerate( chars ) )
#
# # 문자열을 20글자씩 자르고 다음 글자를 등록
# maxlen = 20
# step = 3
# sentences = []
# next_chars = []
# for i in range( 0, len(text)-maxlen, 3 ) :
    # sentences.append( text[i: i+maxlen] )   # 20글자씩 자르고 3글자 이동
    # next_chars.append( text[i+maxlen] )     # 20글자 바로 다음 글자
    #
# import numpy as np    
# train = np.zeros( ( len(sentences), maxlen, len( chars ) ), dtype=np.bool )  
# label = np.zeros( ( len(sentences), len(chars) ), dtype=np.bool )
# print( train.shape )        # (112164, 20, 1483)
# print( label.shape )        # (112164, 1483)
                     #
# for i, sentence in enumerate( sentences ) :     # 112164
    # for j, char in enumerate( sentence ) :      # 20
        # train[ i, j, char_indexs[ char ] ] = 1
    # label[ i, char_indexs[ next_chars[i] ] ] = 1
    #
# from tensorflow.keras import models, layers, optimizers
# model = models.Sequential()
# model.add( layers.LSTM( 128, input_shape=( maxlen, len(chars) ) ) ) # 20행 1483열
# model.add( layers.Dense( len( chars ) ) )       # 1483
# model.add( layers.Activation( "softmax" ))
# optimizer = optimizers.RMSprop( lr=0.01 )
# model.compile( loss="categorical_crossentropy", optimizer=optimizer )
# model.fit( train, label, batch_size=128, epochs=3, validation_split=0.2 )
#
# def sample( predict, temperature=1.0 ) :
    # predict = np.array( predict ).astype( "float64" )
    # predict = np.log( predict ) / temperature
    # exp_predict = np.exp( predict )  
    # pred = exp_predict / np.sum( exp_predict )
    # probas = np.random.multinomial( 1, pred, 1 )    # 확률에 근거해서 실행
    # return np.argmax( probas )       
    #
# import random
# for i in range( 1, 10 ) :
    # print( "-" * 50 )
    # print( i, "번째" )
    # start_index = random.randint( 0, len(text)-maxlen -1 )  # 112164 중에
    # for diversity in [0.2, 0.5, 1.0, 1.2 ] :
        # print( "다양성 : ", diversity )
        # sentence = text[ start_index : start_index + maxlen ]
        # print( "시드 : ", sentence )
        #
        # generate = sentence
        # for i in range (100) :      # 100 글자 추가
            # n = np.zeros((1, maxlen, len(chars)))
            # for j, char in enumerate(sentence) :
                # n[0, j, char_indexs[char]] = 1
            # predict = model.predict( n )[0]
            # next_index = sample ( predict, diversity )
            # next_char = indexs_char [next_index]
            # generate += next_char
        # print( "에측 : " , generate)
    # print()
    #
    #
    #
    #
# 텍스트 생성
# https://www.tensorflow.org/text/tutorials/text_generation?hl=ko
from tensorflow import keras
import tensorflow as tf

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
#print(f'Length of text: {len(text)} characters')    # Length of text: 1115394 characters

# Take a look at the first 250 characters in text
#print(text[:250])

vocab = sorted(list(set(text)))
#print( len(vocab))  # 65

ids_from_chars = keras.layers.experimental.preprocessing.StringLookup(\
                vocabulary=vocab, mask_token=None )
chars_from_ids = keras.layers.experimental.preprocessing.StringLookup(\
                vocabulary=ids_from_chars.get_vocabulary(), invert=True, mask_token=None )

import tensorflow as tf
def text_from_ids( ids ) :
    return tf.strings.reduce_join( chars_from_ids( ids ), axis=1 )

all_ids = ids_from_chars( tf.strings.unicode_split( text, "UTF-8" ) )
#print( all_ids ) # tf.Tensor([19 48 57 ... 46  9  1], shape=(1115394,), dtype=int64)

ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
# for ids in ids_dataset.take( 10 ) :
    # print(chars_from_ids ( ids ).numpy().decode("utf-8") )
    
seq_length = 100
examples_per_epoch = len( text )
# print( examples_per_epoch ) # 1115394
sequences = ids_dataset.batch( seq_length + 1 , drop_remainder=True)
# for seq in sequences.take( 2 ) :
    # print(chars_from_ids( seq))
# for seq in sequences.take( 5 ) :
    # print(chars_from_ids( seq).numpy())
    
def split_input_target(sequence):
    input_text = sequence[: -1]
    target_text = sequence[1:]
    return input_text, target_text

#print( split_input_target("Tensorflow")   )        #('Tensorflo', 'ensorflow')
dataset = sequences.map( split_input_target )

BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = ( dataset.shuffle( BUFFER_SIZE)\
            .batch( BATCH_SIZE, drop_remainder=True )\
            .prefetch( tf.data.experimental.AUTOTUNE ) )   
# print( dataset ) 
vocab_size = len( vocab )   # 65
embedding_dim = 256
run_units = 1024
  
class MyModel( keras.Model ) :
    def __init__( self, vocab_size, embedding_dim, run_units ) :
        super().__init__( self )
        self.embedding = keras.layers.Embedding( vocab_size, embedding_dim )
        self.gru = keras.layers.GRU( run_units, return_sequences=True, return_state=True )
        self.dense = keras.layers.Dense( vocab_size )
    def call( self, inputs, states=None, return_state=False, training=False ) :
        x = inputs
        x = self.embedding( x, training=training )
        if states is None :
            states = self.gru.get_initial_state( x )
        x, states = self.gru( x, initial_state=states, training=training )
        x = self.dense( x, training=training )
        if return_state :
            return x, states
        else :
            return x
        
model = MyModel( vocab_size=len( ids_from_chars.get_vocabulary() ),\
                 embedding_dim=embedding_dim, run_units=run_units )        

for input_example_batch, target_example_batch in dataset.take( 1 ) :
    example_batch_predictions = model( input_example_batch )                

loss = tf.losses.SparseCategoricalCrossentropy( from_logits=True )
example_batch_loss = loss( target_example_batch, example_batch_predictions )
mean_loss = example_batch_loss.numpy().mean()
# print( mean_loss )
model.compile( optimizer="adam", loss=loss )

import os
checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join( checkpoint_dir, "ckpt_{epoch}" )
checkpoint_callback = keras.callbacks.ModelCheckpoint(\
            filepath=checkpoint_prefix, save_weights_only=True )

EPOCHS = 1
history = model.fit( dataset, epochs=EPOCHS, \
                     callbacks=[checkpoint_callback])
#print( history.history )  # {'loss': [2.701610565185547, 1.982250690460205, 1.703411340713501]}

class OneStep( tf.keras.Model ):
    def __init__(self, model, chars_from_ids, ids_from_chars,temperature):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars
        
        skip_ids = self.ids_from_chars( ["UNK"] )[:, None]
        sparse_mask  = tf.SparseTensor(\
                                       values = [-float("inf")] * len(skip_ids),
                                       indices = skip_ids,
                                       dense_shape=[len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)
        
    @tf.function
    def generate_one_step(self, inputs, states=None ) :
        input_chars = tf.strings.unicode_split( inputs, "UTF-8" )
        input_ids = self.ids_from_chars( input_chars ).to_tensor()
        
        predicted_logits, states = self.model( \
                inputs=input_ids, states=states, return_state=True )
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits / self.temperature
        predicted_logits = predicted_logits + self.prediction_mask
        
        predicted_ids = tf.random.categorical( predicted_logits, num_samples=1 )
        predicted_ids = tf.squeeze( predicted_ids, axis=1 )
        predicted_chars = self.chars_from_ids( predicted_ids )
        return predicted_chars, states

one_step_model = OneStep( model, chars_from_ids, ids_from_chars )
states = None
next_char = tf.constant( ["ROMEO:"] ) 
result = [next_char]

for n in range( 1000 ) :
    next_char, states = one_step_model.generate_one_step( next_char, \
                            states=states )
    result.append( next_char )
result = tf.strings.join( result )
print( result[0].numpy().decode( "utf-8"), "\n\n" + "_"*80 )

class CustomTraining( MyModel ) :
    @tf.function
    def train_step(self, inputs ) :
        inputs, labels = inputs
        with tf.GradientTape() as tape :
            predictions = self( inputs, training=True )
            loss = self.loss( labels, predictions )
        grads = tape.gradient( loss, model.trainable_variables )
        self.optimizer.apply_gradients( zip( grads, model.trainable_variables ) )
        return {"loss":loss}
model = CustomTraining( vocab_size=len( ids_from_chars.get_vocabulary() ),\
                        embedding_dim = embedding_dim, run_units=run_units )
model.compile( optimizer=keras.optimizers.Adam(), 
               loss=keras.losses.SparseCategoricalCrossentropy( from_logits=True ) )
# history = model.fit( dataset, epochs=EPOCHS )
# print( history.history )

EPOCHS = 5
mean = tf.metrics.Mean()
for epoch in range( EPOCHS ) :
    mean.reset_states()
    for (batch_n, (inp, target) ) in enumerate( dataset ) :
        logs = model.train_step( [inp, target] )
        mean.update_state( logs["loss"] )
        if batch_n % 50 == 0 :
            template = f"Epoch {epoch+1} Batch {batch_n} Loss {logs['loss']:.4f}"
            print( template )
    
    print()
    print( f"Epoch {epoch+1} Loss : {mean.result().numpy():.4f}")
    print( "_"*80 )
        
