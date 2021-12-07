# 자연어 처리

# samples = [
    # "내가 그의 이름을 불러주기 전에는",
    # "그는 다만",
    # "하나의 몸짓에 지나지 않았다.",
    # "내가 그의 이름을 불러주었을 때,",
    # "그는 나에게로 와서",
    # "꽃이 되었다.",
    # "내가 그의 이름을 불러준 것처럼",
    # "나의 이 빛깔과 향기에 알맞은",
    # "누가 나의 이름을 불러다오.",
    # "그에게로 가서 나도",
    # "그의 꽃이 되고 싶다.",
    # "우리들은 모두",
    # "무엇이 되고 싶다.",
    # "너는 나에게 나는 너에게",
    # "잊혀지지 않는 하나의 눈짓이 되고 싶다."]
# label = [[1], [0], [1], [0], [1], [0], [0], [1], \
         # [0], [0], [1], [0], [0], [1], [1]] 
         #
# import tensorflow as tf
# from tensorflow.keras import preprocessing
#
# token = preprocessing.text.Tokenizer()
# token.fit_on_texts( samples )           # 단어에 인덱스 부여
# sequences = token.texts_to_sequences( samples )
# # print( sequences )
# max_len = 7
# sequences = preprocessing.sequence.pad_sequences( \
                # sequences, maxlen=max_len )
# # print( sequences )
#
# # print( token.word_index )
# dataset = tf.data.Dataset.from_tensor_slices( (sequences, label ) )
# # print( dataset )
#
# dataset = dataset.batch( 2 )
# dataset = dataset.shuffle( len( sequences ) )
# dataset = dataset.repeat( 2 )
# dataset = dataset.map( lambda x, y : ({"x":x}, y ) ) 
#
# iterator = tf.compat.v1.data.make_one_shot_iterator( dataset )
# for word in iterator :
    # print( word )


# Word2Vec
# https://www.tensorflow.org/tutorials/text/word2vec?hl=ko

# 형태소 분석
# from konlpy.tag import Okt
# okt = Okt()
# words = okt.pos( "아버지가 방에 들어 가신다. ㅎㅎ", norm=True, stem=True )
# print( words )
# words = okt.pos( "아버지가 방에 들어 가신다. ㅎㅎ" )
# print( words )

# 명사 빈도 분석
import codecs
with codecs.open( "토지1.txt", "r", encoding="ms949" ) as f :
    data = f.read()
# print( data )
# print( type( data ) )       # str

# from konlpy.tag import Okt
# from konlpy import jvm
# jvm.init_jvm()
# okt = Okt()
# lines = data.split( "\r\n" )
# word_dict = {}
# for line in lines :
    # words = okt.pos( line )
    # for word, pumsa in words :
        # # print( word, pumsa )
        # if pumsa == "Noun" :
            # if not word in word_dict :
                # word_dict[word] = 0
            # else :
                # word_dict[word] += 1 
# sorted_word = sorted( word_dict.items(), key=lambda x:x[1], reverse=True )
# for word, count in sorted_word[:50] :
    # print( word, " : ", count )


# 토지1.txt Word2Vec 모델 생성
import codecs
from konlpy.tag import Okt
with codecs.open( "토지1.txt", "r", encoding="ms949" ) as f :
    data = f.read()
okt= Okt()
lines = data.split( "\r\n" )
words = []
for line in lines :
    word = []
    word_list = okt.pos( line, norm=True, stem=True )
    for w, p in word_list :
        if not p in ["Josa", "Eomi", "Punctuation"] :
            word.append( w )
    words.append( (" ".join( word )).strip() )            
output = ( " ".join( words )).strip()
# print( output )

with open( "toji.dat", "w", encoding="utf-8" ) as f :
    f.write( output )

from gensim.models import word2vec, Word2Vec
data = word2vec.LineSentence( "toji.dat" )
model = Word2Vec( data, vector_size=200, window=10, hs=1, min_count=2, sg=1 )
model.save( "toji.model" )    
    
model = Word2Vec.load("toji.model")
model.train(data,total_examples=1, epochs=1)

# print( model.wv.most_similar( positive=["땅"] ) )
# print( model.wv.most_similar( positive=["이슬"] ) )
# print( model.wv.most_similar( positive=["땅","이슬"],negative=["바람"] ) )
# print( model.wv.most_similar( positive=["거구"], negative=["깨닫"]))

del model

















