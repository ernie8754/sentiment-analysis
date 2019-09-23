# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 11:24:12 2019

@author: zxc63
"""
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing import sequence
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
        check=False
        for word in words:
            if word==',':
                check=False
                continue
            if word !='n\'t':
                if not word.isdigit():
                    if check:
                        word='not_'+word
                    word_freqs[word] += 1
            else:
                check=True
        num_recs += 1
## 準備數據
MAX_FEATURES = 16000
max_length = 50
vocab_size = min(MAX_FEATURES, len(word_freqs)) + 2

####===================================================================================================
word_index = {x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))}
word_index["PAD"] = 0
word_index["UNK"] = 1
X = []                          #word2vec training set
X_train_data = np.empty(num_recs,dtype=list)               #訓練集
y = np.zeros(num_recs)                        #訓練集的Label
data_test = []                  #測試集
data_test_label = []            #測試集的Label
word_dict={}                    #詞向量轉換

with open('5-label-train.csv', newline = '',encoding = 'UTF8') as csvfile :
    i=0
    rows = csv.DictReader(csvfile)
    for row in rows:
        sentence = row['Phrase']
        label = row['Sentiment']
        sentence=sentence.translate(table)
        words = nltk.word_tokenize(sentence.lower())
        seqs = []
        check=False;
        for word in words:
            if word==',':
                check=False
                continue
            if word !='n\'t':
                if not word.isdigit():
                    if check:
                        word='not_'+word
                    if word in word_index:
                        seqs.append(word_index[word])
                    else:
                        seqs.append(word_index["UNK"])
            else:
                check=True
        X_train_data[i] = seqs
        y[i] = int(label)
        i += 1


X_train_data = sequence.pad_sequences(X_train_data, maxlen=maxlen)
y = to_categorical(np.asarray(y))
X_train, Xtest, y_train, ytest = train_test_split(X_train_data, y, test_size=0.2, random_state=42)
X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int8)


Xtest = np.array(Xtest, dtype=np.float32)
ytest = np.array(ytest, dtype=np.int8)
input_shape = (maxlen,100,)
BATCH_SIZE = 100
NUM_EPOCHS = 20
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_SIZE,input_length=maxlen))
model.add(Conv1D(filters=32, kernel_size = 3, padding='same', activation='relu',input_shape = input_shape))
model.add(MaxPooling1D(pool_size = 2))
model.add(Dropout(0.2))
# 1 layer of 100 units in the hidden layers of the LSTM cells
model.add(LSTM(64))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 模型訓練
checkpoint = ModelCheckpoint( 'sentiment-analysis-model',verbose=1, monitor='loss',save_best_only = True, mode='min')
early_stopping =  EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto', baseline=None)

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

csvfile = open('5-label-test.csv', newline = '',encoding = 'UTF8')
rows = csv.DictReader(csvfile)
data_test_label = []
data_test=[]
for row in rows:
    sentence = row['Phrase']
    label = row['PhraseId']
    data_test_label.append(label)
    data_test.append(sentence)
X = np.empty(len(data_test),dtype=list)
csvfile.close()
with open('output.csv', 'w', newline='') as csvfile:
    i=0
    for sentence in  data_test:
        sentence=sentence.translate(table)
        words = nltk.word_tokenize(sentence.lower())
        seqs = []
        check=False;
        for word in words:
            if word==',':
                check=False
                continue
            if word !='n\'t':
                if not word.isdigit():
                    if check:
                        word='not_'+word
                    if word in word_index:
                        seqs.append(word_index[word])
                    else:
                        seqs.append(word_index["UNK"])
            else:
                check=True
        X[i] = seqs
        i+=1
    X = sequence.pad_sequences(X, maxlen=maxlen)
    labels = [np.argmax(x) for x in model.predict(X) ]
    #print(labels)
  # 以空白分隔欄位，建立 CSV 檔寫入器
  #data_test = np.array(data_test, dtype=np.float32)
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['PhraseId', 'Sentiment'])
    for i,x in enumerate(labels):
        writer.writerow([data_test_label[i], labels[i]])