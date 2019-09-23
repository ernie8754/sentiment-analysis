# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 15:46:37 2019

@author: zxc63
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM,Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from gensim.models import Word2Vec
import collections
import nltk
import numpy as np
import csv
import matplotlib.pyplot as plt
import string


maxlen = 0
word_freqs = collections.Counter()
num_recs = 0
punc=string.punctuation
punc=punc.replace('\'','')
punc=punc.replace(',','')
table = str.maketrans({key: None for key in punc})
with open('5-label-train.csv', newline = '',encoding = 'UTF8' ) as csvfile :
    rows = csv.DictReader(csvfile)
    for row in rows:
        sentence = row['Phrase']
        label = row['Sentiment']
        sentence=sentence.translate(table)
        words = nltk.word_tokenize(sentence.lower())
        if len(words) > maxlen:
            maxlen = len(words)
        for word in words:
            if not word.isdigit():
                word_freqs[word] += 1
        num_recs += 1  
## 準備數據
MAX_FEATURES = 16000
max_length = 50
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2

####===================================================================================================

X = []                          #word2vec training set
X_train_data = []               #訓練集
y = []                          #訓練集的Label
data_test = []                  #測試集              
data_test_label = []            #測試集的Label
word_dict={}                    #詞向量轉換

with open('5-label-train.csv', newline = '',encoding = 'UTF8') as csvfile :
    rows = csv.DictReader(csvfile)
    for row in rows:
        sentence = row['Phrase']
        label = row['Sentiment']
        sentence=sentence.translate(table)
        words = nltk.word_tokenize(sentence.lower())
        n_words=[]
        for word in words:
            if not word.isdigit():
                n_words.append(word)
        X.append(n_words)
        X_train_data.append(n_words)
        y.append(int(label))
csvfile = open('5-label-test.csv', newline = '',encoding = 'UTF8')
rows = csv.DictReader(csvfile)
for row in rows:
    label = row['PhraseId']
    sentence = row['Phrase']
    sentence=sentence.translate(table)
    words = nltk.word_tokenize(sentence.lower()) 
    n_words=[]
    for word in words:
        if not word.isdigit():
            n_words.append(word)    
    data_test.append(n_words)
    X.append(n_words)
    data_test_label.append(label)
csvfile.close()


EMBEDDING_DIM = 100

w2v_model = Word2Vec(X,size=EMBEDDING_DIM,min_count=5,workers=2,iter=50) #訓練詞向量
for k, v in w2v_model.wv.vocab.items(): #載入詞向量的權重
	word_dict[k] = w2v_model.wv[k]
del w2v_model

print(1)

for idx in range(len(X_train_data)):
    tmp=[]
    for word in X_train_data[idx]:
        if word in word_dict:
            tmp.append(np.float32(word_dict[word]))
        else:
            tmp.append(np.zeros(EMBEDDING_DIM,dtype=np.float32))
    if len(tmp)<max_length:
        for rest in range(max_length-len(tmp)):
            tmp.append(np.zeros(EMBEDDING_DIM,dtype=np.float32))
    elif len(tmp)>max_length:
        tmp=tmp[:max_length]
    X_train_data[idx] = tmp

for idx in range(len(data_test)):
    tmp=[]
    for word in data_test[idx]:
        if word in word_dict:
            tmp.append(np.float32(word_dict[word]))
        else:
            tmp.append(np.zeros(EMBEDDING_DIM,dtype=np.float32))
    if len(tmp)<max_length:
        for rest in range(max_length-len(tmp)):
            tmp.append(np.zeros(EMBEDDING_DIM,dtype=np.float32))
    elif len(tmp)>max_length:
        tmp=tmp[:max_length]
    data_test[idx] = tmp

print(2)

y = to_categorical(np.asarray(y))

# 資料劃分訓練組及測試組
X_train, Xtest, y_train, ytest = train_test_split(X_train_data, y, test_size=0.2, random_state=42)

del word_freqs
del word_dict
del X
del y
del X_train_data

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int8)


Xtest = np.array(Xtest, dtype=np.float32)
ytest = np.array(ytest, dtype=np.int8)

# 模型構建
print(3)

input_shape = (X_train.shape[1],X_train.shape[2],)
BATCH_SIZE = 100
NUM_EPOCHS = 20
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64
"""embedding_layer = Embedding(vocab_size + 1 ,
                            EMBEDDING_DIM,
                            weights = word_dict,
                            input_length = max_length,
                            trainable=False)"""
model = Sequential()
model.add(Conv1D(filters=32, kernel_size = 3, padding='same', activation='relu',input_shape = input_shape))
model.add(MaxPooling1D(pool_size = 2))
model.add(Dropout(0.2))
# 1 layer of 100 units in the hidden layers of the LSTM cells
model.add(LSTM(64))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 模型訓練
checkpoint = ModelCheckpoint( 'sentiment-analysis-model',verbose=1, monitor='loss',save_best_only = True, mode='min')
early_stopping =  EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None)

history = model.fit(X_train, y_train, batch_size=BATCH_SIZE,  callbacks=[checkpoint,early_stopping], epochs=NUM_EPOCHS,validation_data=(Xtest, ytest))

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# 預測
#score, acc = model.evaluate(Xtest, ytest, batch_size=BATCH_SIZE)
#print("\nTest score: %.3f, accuracy: %.3f" % (score, acc))
#print('{}   {}      {}'.format('預測','真實','句子'))
#for i in range(5):
#    idx = np.random.randint(len(Xtest))
#    xtest = Xtest[idx].reshape(1,MAX_SENTENCE_LENGTH)
#    ylabel = ytest[idx]
#    ypred = np.argmax(model.predict(xtest),axis = 1)
#    sent = " ".join([index2word[x] for x in xtest[0] if x != 0])
#    print(' {}      {}     {}'.format(np.reshape(ypred,(1)), int(ylabel), sent))

with open('output.csv', 'w', newline='') as csvfile:

  # 以空白分隔欄位，建立 CSV 檔寫入器
  data_test = np.array(data_test, dtype=np.float32)
  writer = csv.writer(csvfile, delimiter=',')
  writer.writerow(['PhraseId', 'Sentiment'])
  X = np.argmax(model.predict(data_test),axis = 1)
  for i, x in enumerate(X): 
      writer.writerow([data_test_label[i], x ])