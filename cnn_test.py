# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import tensorflow as tf
import time
import cv2
import os

from keras.utils import np_utils     #匯入 Keras 的 Numpy 工具 
import numpy as np                       
np.random.seed(10)           #設定隨機種子, 以便每次執行結果相同
from keras.datasets import mnist    #匯入 mnist 模組後載入資料集
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D 

df0 = pd.DataFrame()
print(df0)



(x_train_image, y_train_label), (x_test_image, y_test_label)=mnist.load_data()
print(x_test_image.shape)
x_train = x_train_image.reshape(60000,28,28,1).astype('float32') 
x_test = x_test_image.reshape(10000,28,28,1).astype('float32') 
print(x_test[0].shape)

x_train_normalize = x_train/255
x_test_normalize = x_test/255
print(y_test_label)
y_train_onehot=np_utils.to_categorical(y_train_label)
y_test_onehot=np_utils.to_categorical(y_test_label)
print(y_test_onehot)
label_num = len(y_train_onehot[0])


#print(os.environ["CUDA_VISIBLE_DEVICES"])
# 切換使用cpu
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #为使用CPU  
print('gpu使用', tf.test.is_gpu_available())
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())



# 模型建立
model = Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = (28,28,1)))
model.add(MaxPooling2D(pool_size = (2,2)))
# Add another:
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(Dense(label_num, activation='softmax'))
#print(model.summary())       
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
t0 = time.time()
train_history=model.fit(x=x_train_normalize,
                        y=y_train_onehot,validation_split=0.2, 
                        epochs=2, batch_size=30,verbose=2)


"""
x=正規化後的 28*28 圖片特徵向量
y=One-hot 編碼的圖片標籤 (答案)
validation_split=驗證資料集占訓練集之比率
epochs=訓練週期
batch_size=每一批次之資料筆數
verbose=顯示選項 (2=顯示訓練過程)
"""

t1 = time.time()
print(int(t1-t0), 's')