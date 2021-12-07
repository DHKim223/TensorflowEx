# Word2Vec
# 영화 리뷰 분석
# f = open( "ratings.csv", encoding="ms949" )
# data = f.read()
# # print( type( data ) )               # str

# import pandas as pd
# data = pd.read_csv( "ratings_test.csv", encoding="ms949" )
# data = data["document"]
#
# from konlpy.tag import Okt
# okt = Okt()
#
# results = []
# for line in data :
    # line = str( line )
    # r = []
    # word_list = okt.pos( line, norm=True, stem=True )
    # for word, pumsa in word_list :
        # if not pumsa in ["Josa", "Eomi", "Punctuation"] :
            # r.append( word )
    # results.append( (" ".join( r )).strip() )
# output = ( " ".join( results )).strip()
# with open( "review.dat", "w", encoding="utf-8" ) as f :
    # f.write( output )   
    #
# from gensim.models import word2vec
# data = word2vec.LineSentence( "review.dat" )    
# model = word2vec.Word2Vec( data, vector_size=100, window=5, hs=1, \
                           # min_count=2, sg=1 )
# model.save( "review.model" )
# model = word2vec.Word2Vec.load( "review.model" )
#
# print( model.wv.most_similar( positive=["영화"] ) )
# print( model.wv.most_similar( positive=["드라마"] ) )
#
# print( model.wv.most_similar( "영화", "드라마" ) )   # 키워드간의 유사도
# print( model.wv.most_similar( "영화", "액션" ) )
# print( model.wv.doesnt_match( "영화 드라마 이야기 액션".split() ) )
#
# from wordcloud import WordCloud
# words = open( "ratings_test.csv", encoding="ms949" ).read()
# wc = WordCloud( background_color="white", \
                # font_path="c:\Windows\Fonts\malgun.ttf" ).generate( words )
# wc.to_file( "review.png" )
#
# import matplotlib.pyplot as plt
# plt.figure( figsize=( 5, 5 ) )
# plt.imshow( wc, interpolation="bilinear" )
# plt.axis( "off" )
# plt.show()              



# IMDB
# 긍정 25000  부정 25000        5:5
from tensorflow.keras.datasets import imdb
(train_data, train_label), (test_data, test_label ) \
    = imdb.load_data( num_words=10000 )
# print( train_data.shape )           # (25000,)
# print( train_label.shape )          # (25000,)

# print( train_data )
# print( train_label )
print( max( [ max( data ) for data in train_data ] ) )  # 88586

word_index = imdb.get_word_index()
# print( list( word_index.items() )[:10] )

# 데이터를 one hot encoding
import numpy as np
def one_hot_encoding( datas, demension=10000 ) :
    results = np.zeros( ( len( datas ), demension ) )
    for i, data in enumerate( datas ) :
        results[i, data] = 1
    return results     
train_data = one_hot_encoding( train_data )
test_data = one_hot_encoding( test_data )
# print( train_data.shape )       # (25000, 10000)
# print( test_data.shape )        # (25000, 10000)
train_label = np.asarray( train_label ).astype( "float32" )
test_label = np.asarray( test_label ).astype( "float32" )
# print( train_label.shape )      # (25000,)
# print( test_label.shape )       # (25000,)

from tensorflow.keras import models, layers
model = models.Sequential()
model.add( layers.Dense( 16, activation="relu", input_shape=(10000,) ) )
model.add( layers.Dense( 16, activation="relu" ) )
model.add( layers.Dense( 1, activation="sigmoid") )
model.compile( optimizer="rmsprop", loss="binary_crossentropy", \
               metrics=["accuracy"] )
hist = model.fit( train_data, train_label, epochs=5, batch_size=512, \
                  validation_data=(train_data, train_label) )
score = model.evaluate( test_data, test_label )
print( score )
print( model.predict( test_data )[:10] )
print( test_label[:10] )
predicts = model.predict( test_data )

tzero = ( train_label == 0 ).sum()
tone = ( train_label == 1 ).sum()
predicts = np.array( [ 0 if i<0.5 else 1 for i in predicts ] )
pzero = ( predicts == 0 ).sum()
pone = ( predicts == 1 ).sum()

import matplotlib.pyplot as plt
plt.bar( np.arange( 2 ), ( pzero, pone ), color="blue", alpha=0.3 )
plt.bar( np.arange( 2 ), ( tzero, tone ), color="red", alpha=0.3 )
plt.xticks( [0, 1] )
plt.yticks( [i for i in range( 0, len( train_label )//2, 1000 ) ] )
plt.xlabel( "label" )
plt.ylabel( "count" )
plt.show()                   















    
    
    
    
    
    
    