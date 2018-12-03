#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 01:18:42 2018

@author: vuchicago
"""
import os
os.chdir("/Users/vuchicago/Documents/Courses/Python Class/deep_learning_files")
import numpy as np
import keras
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions

resnet50=ResNet50(weights='imagenet')

pic="panda.jpg"
pic="pikachu.png"
pic='fox.png'
pic='kitten.jpg'
pic='grumpycat.jpg'
pic='kitten2.jpg'
img=image.load_img(pic,target_size=(224,224))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)

preds=resnet50.predict(x)
predict=decode_predictions(preds,top=2)
print("Neural Net predicts this is a: ",predict[0][0])
print("Neural Net predicts this is a: ",predict[0][1])



########
########
#imdb data of 50k reviews.  Train and test set are 25k each.  Each set around 50% postiive and 50% negative reviews
from keras.datasets import imdb
(train_data,train_labels),(test_data,test_labels)=imdb.load_data(num_words=10000) #num_words is 10k most frequent words.  All others will be discarded

max([max(sequence) for sequence in train_data])


word_index=imdb.get_word_index()


###get the word to the index.  
#Subtract 3 because first 3 are reserved for 'padding','start of sequence', and 'unknown'
reverse_word_index=dict(
        [(value,key) for (key,value) in word_index.items()])
decoded_reivew=' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])

#PREP THE DATA

import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results=np.zeros((len(sequences),dimension)) ##(25K X 10000)
    for i, sequence in enumerate(sequences):
        results[i,sequence]=1

x_train=vectorize_sequences(train_data)
x_label=vectorize_sequences(test_data)

y_train=np.asarray(train_labels).astype('float32') #turn labbels into an array
y_test=np.asarray(test_labels).astype('float32')
