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

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from keras.applications import Xception
from keras.layers import Input, Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam




(x_train_image, y_train_label), (x_test_image, y_test_label)=mnist.load_data()
print(x_test_image.shape)

mouse_path = r'D:\python\Mouse'
data_list = []
label_list = []
label_count = 0
img_size = 200
for mouse_day in os.listdir(mouse_path):
    imgs_path = os.path.join(mouse_path, mouse_day)
    for img_name0 in os.listdir(imgs_path):
        img_path0 = os.path.join(imgs_path, img_name0)
        img0 = cv2.imread(img_path0)
        img0 = cv2.resize(img0,(img_size,img_size))
        data_list.append(img0)
        label_list.append(label_count * np.ones(1, np.uint8))
    label_count += 1
img_datas = np.array(data_list)
img_labels = np.array(label_list)
#x_train = x_train_image.reshape(60000,28,28,1).astype('float32') 
#x_test = x_test_image.reshape(10000,28,28,1).astype('float32') 
seed = 7
x_train,x_test,y_train,y_test = train_test_split(img_datas,img_labels,test_size=0.1,random_state=seed)


print(x_test[0].shape)

x_train_normalize = x_train/255
x_test_normalize = x_test/255
print(y_test_label)
y_train_onehot=np_utils.to_categorical(y_train)
y_test_onehot=np_utils.to_categorical(y_test)
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
model.add(Conv2D(64,(3,3), activation = 'relu', input_shape = (img_size,img_size,3)))
model.add(MaxPooling2D(pool_size = (2,2)))
# Add another:
model.add(Conv2D(64,(3,3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
# Add a softmax layer with 10 output units:
model.add(Dense(label_num, activation='softmax'))
    
model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])






# xception
model = Xception(include_top=False,
                 weights='imagenet',
                 input_tensor=Input(shape=(img_size, img_size, 3))
                 )

# 定義輸出層
x = model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(label_num, activation='softmax')(x)
model = Model(inputs=model.input, outputs=predictions)

# 編譯模型
model.compile(optimizer=Adam(lr=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])




print(model.summary())   




t0 = time.time()








train_history=model.fit(x=x_train_normalize,
                        y=y_train_onehot,validation_split=0.2, 
                        epochs=5, batch_size=30,verbose=2)


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

pred_path = r"D:\python\Mouse\7day\7_3.bmp"
img0 = cv2.imread(pred_path)
img0 = cv2.resize(img0,(img_size,img_size))
img0 = img0/255
img0 = img0.reshape(1,200,200,3).astype('float32') 
ans00 = model.predict(img0)
ans0 = np.argmax(ans00,axis=1)
#ans0 = model.predict_classes(img0)
print(ans0)
print("---")
path0 = r"D:\python\Mouse\7day"
test_data0 = []
for i in os.listdir(path0):
    pred_path = os.path.join(path0, i)
    img0 = cv2.imread(pred_path)
    img0 = cv2.resize(img0,(img_size,img_size))
    img0 = img0/255
    test_data0.append(img0)
    img0 = img0.reshape(1,200,200,3).astype('float32') 
    ans00 = model.predict(img0)
    ans0 = np.argmax(ans00,axis=1)
    #ans0 = model.predict_classes(img0)
    print(ans0)


test_data = np.array(test_data0)
ans00 = model.predict(test_data)
ans0 = np.argmax(ans00,axis=1)
print(ans0)