# https://www.tensorflow.org/tutorials/text/word2vec?hl=ko

import tensorflow as tf
SEED = 42
AUTOTUNE = tf.data.AUTOTUNE

sentence = "The wide road shimmered in the hot sun"
tokens = list(sentence.lower().split() )
#print(len(tokens))  #8

vocab, index = {}, 1
vocab["<pad>"] = 0
for token in tokens : 
    if token not in vocab :
        vocab[token] = index
        index += 1
vocab_size = len(vocab)
#print(vocab) 
# {'<pad>': 0, 'the': 1, 'wide': 2, 'road': 3, 'shimmered': 4, 'in': 5, 'hot': 6, 'sun': 7}
inverse_vocab = {index : token for token, index in vocab.items()}
#print( inverse_vocab )
# {0: '<pad>', 1: 'the', 2: 'wide', 3: 'road', 4: 'shimmered', 5: 'in', 6: 'hot', 7: 'sun'}

example_sequence = [vocab[word] for word in tokens]
# print( example_sequence)
# [1,2,3,4,5,1,6,7]
window_size = 2
positive_skip_grams , _ = tf.keras.preprocessing.sequence.skipgrams(
    example_sequence,
    vocabulary_size = vocab_size,
    window_size = window_size,
    negative_samples = 0 )
# print( len(positive_skip_grams )) # 26
#print(positive_skip_grams)
# [[3, 2], [3, 1], [2, 3], [2, 4], [6, 1], [3, 5], [4, 3], [5, 3], [1, 6], [4, 5], 
# [1, 7], [1, 2], [1, 5], [2, 1], [7, 1], [4, 2], [4, 1], [1, 3], [3, 4], [5, 1], [7, 6], [5, 4], [6, 7], [5, 6], [6, 5], [1, 4]]

target_word, context_word = positive_skip_grams[0]
num_ns = 4
context_class = tf.reshape( tf.constant ( context_word, dtype="int64"), ( 1,1))
#print( context_class) #  tf.Tensor([[3]], shape=(1, 1), dtype=int64)
negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
    true_classes = context_class,
    num_true = 1,
    num_sampled = num_ns,
    unique = True,
    range_max = vocab_size,
    seed = SEED, 
    name = "negative_sampling")
# print(negative_sampling_candidates) # tf.Tensor([2 1 4 3], shape=(4,), dtype=int64)
# print( inverse_vocab[ 6 ] )         # hot
# print( [inverse_vocab[index.numpy( ) ] \
         # for index in negative_sampling_candidates] ) # ['wide', 'the', 'shimmered', 'road']

negative_sampling_candidates = tf.expand_dims(negative_sampling_candidates , 1 )
context = tf.concat( [context_class, negative_sampling_candidates] , 0 )
label = tf.constant( [1] + [0]*num_ns, dtype="int64")
            # [ 1, 0, 0, 0, 0 ]
target = tf.squeeze(target_word)
context = tf.squeeze(context)
label = tf.squeeze( label )
# print( f"target_index : {target}")
# print(f"target_word : {inverse_vocab[target_word]}")
# print(f"context_indices : {context}")
# print(f"context_words : {[inverse_vocab[c.numpy()] for c in context ]}")
# print(f"label : {label}")
# target_index : 6
# target_word : hot
# context_indices : [1 2 1 4 3]
# context_words : ['the', 'wide', 'the', 'shimmered', 'road']
# label : [1 0 0 0 0]

import tqdm
def generate_training_data( sequences, window_size, num_ns, vocab_size, seed ):
    targets, contexts, labels = [], [], []
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table( vocab_size)
    for sequence in tqdm.tqdm(sequences):
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size = vocab_size,
            sampling_table = sampling_table,
            window_size = window_size,
            negative_samples = 0)
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(tf.constant( [context_word], \
                                                        dtype="int64"), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(\
                                                                                         true_classes = context_class,
                                                                                         num_true = 1,
                                                                                         num_sampled = num_ns,
                                                                                         unique = True,
                                                                                         range_max = vocab_size,
                                                                                         seed = seed,
                                                                                         name = "negative_sampling")
            negative_sampling_candidates = tf.expand_dims( negative_sampling_candidates , 1)
            context = tf.concat( [context_class, negative_sampling_candidates], 0 )
            label = tf.constant([1] + [0]*num_ns, dtype="int64")
            
            targets.append( target_word)
            contexts.append(context)
            labels.append(label)
    return targets, contexts, labels

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
# with open ( path_to_file) as f :
    # lines = f.read().splitlines()
# for line in lines [:20] :
    # print(line)
    
text_ds = tf.data.TextLineDataset(path_to_file).filter(lambda x: tf.cast(tf.strings.length(x), bool))

import re
import string
from tensorflow.keras import layers
def custom_standardization( input_data ):
    lowercase = tf.strings.lower(input_data)
    return tf.strings.regex_replace(lowercase, "[%s]" %re.escape(string.punctuation), '')
vocab_size = 4096
sequence_length = 10
vectorize_layer = layers.experimental.preprocessing.TextVectorization(      # 낮은 버전용
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length
    )
vectorize_layer.adapt(text_ds.batch(1024))
inverse_vocab = vectorize_layer.get_vocabulary()
#print(inverse_vocab[:20])
text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map( \
                                                             vectorize_layer).unbatch()
sequences = list(text_vector_ds.as_numpy_iterator())                                                         
#print(sequences)
for seq in sequences[:5] :
    print(f"{seq} => {[inverse_vocab[i] for i in seq]}")
    
targets, contexts, labels = generate_training_data(
    sequences = sequences,
    window_size = 2,
    num_ns =4,
    vocab_size = vocab_size,
    seed = SEED)

import numpy as np
targets = np.array(targets)
contexts = np.array(contexts)[:, :, 0]
labels = np.array( labels)
# print(targets.shape) # (65436,)
# print(contexts.shape) # (65436, 5)
# print(labels.shape) # (65436, 5)
print(targets)
print(contexts)
print(labels)

# [1286 1286 1286 ... 2050 1402 1402]

# [[ 344  170   19  826   57]
 # [  34    1   20   84   11]
 # [   4  280 1633    3    9]
 # ...
 # [ 186  234    0 1303   50]
 # [ 674  444    1    9   23]
 # [  28  364   77   14   28]]
 
# [[1 0 0 0 0]
 # [1 0 0 0 0]
 # [1 0 0 0 0]
 # ...
 # [1 0 0 0 0]
 # [1 0 0 0 0]
 # [1 0 0 0 0]]

BATCH_SIZE = 1024
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)
# print(dataset) 
# <PrefetchDataset element_spec=((TensorSpec(shape=(1024,), dtype=tf.int64, name=None), TensorSpec(shape=(1024, 5), dtype=tf.int64, name=None)), TensorSpec(shape=(1024, 5), dtype=tf.int64, name=None))>
class Word2Vec(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim):
    super(Word2Vec, self).__init__()
    self.target_embedding = layers.Embedding(\
                                             vocab_size, embedding_dim,  input_length=1, name="w2v_embedding")
    self.context_embedding = layers.Embedding(\
                                              vocab_size, embedding_dim, input_length=num_ns+1)

  def call(self, pair):
    target, context = pair
    # target: (batch, dummy?)  # The dummy axis doesn't exist in TF2.7+
    # context: (batch, context)
    if len(target.shape) == 2:
      target = tf.squeeze(target, axis=1)
    # target: (batch,)
    word_emb = self.target_embedding(target)
    # word_emb: (batch, embed)
    context_emb = self.context_embedding(context)
    # context_emb: (batch, context, embed)
    dots = tf.einsum('be,bce->bc', word_emb, context_emb)
    # dots: (batch, context)
    return dots #( batch, context )

def custom_loss(x_logit, y_true):
      return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)
  
embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(optimizer='adam', \
                 loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), \
                 metrics=['accuracy'])

word2vec.fit( dataset, epochs = 20 )

weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()
import io
out_v = io.open("vectors.tsv", "w", encoding="utf-8")
out_w = io.open("metadata.tsv","w",encoding="utf-8")

out_v = io.open('vectors.tsv', 'w', encoding='utf-8')
out_m = io.open('metadata.tsv', 'w', encoding='utf-8')

for index, word in enumerate(vocab):
    if index == 0:
        continue  # skip 0, it's padding.
    vec = weights[index]
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
    out_m.write(word + "\n")
out_v.close()
out_m.close()