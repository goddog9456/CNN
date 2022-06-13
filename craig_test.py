# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 14:25:36 2022

@author: L7A
"""

# -*- coding: utf-8 -*-
from tensorflow import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
import os

import seaborn as sns
from keras import optimizers
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix

import tensorflow as tf
import cv2



import matplotlib.pyplot as plt



from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import  Adam, SGD


# image_path:影像路徑(單張)
# model: CNN Model物件(訓練好的)
# last_layer_name: CONV最後一層的名稱   可用model.summary()查看
def grad_cam(image_path, model, last_layer_name):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH))
    img = tf.keras.preprocessing.image.img_to_array(img)
    
    
    #model = tf.keras.applications.resnet.ResNet50(weights='imagenet', include_top=True)
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_layer_name).output, model.output])
      
    
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.array([img]))
        loss = predictions[:, np.argmax(predictions[0])]

    output = conv_outputs[0]
    grads = tape.gradient(loss, conv_outputs)[0]

    gate_f = tf.cast(output > 0, 'float32')
    gate_r = tf.cast(grads > 0, 'float32')
    guided_grads = tf.cast(output > 0, 'float32') * tf.cast(grads > 0, 'float32') * grads

    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    cam = np.ones(output.shape[0: 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam.numpy(), (IMAGE_WIDTH , IMAGE_HEIGHT ))
    cam = np.maximum(cam, 0)
    heatmap = (cam - cam.min()) / (cam.max() - cam.min())

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_RAINBOW)

    output_image = cv2.addWeighted(cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR), 1, cam, 0.5, 0)

    return output_image


# 路徑
TRAIN_DIR = r'./train-2type'
VAL_DIR = r'./val-2type'
FIG_PATH = r'./Fig'
MODEL_PATH = r'./Model'

# 參數
IMAGE_WIDTH = 175  # 50" 200    43" 175
IMAGE_HEIGHT = 350 # 50" 400    43" 350 
LR = 5e-5          #5e-5
EPOCHS = 15
BATCH_SIZE = 16
CLASSES=2



if __name__ == '__main__':
    """
    # Scale 操作
    train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)     #############
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # 使用迭代器生成圖片張量
    train_gen = train_datagen.flow_from_directory(TRAIN_DIR, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=BATCH_SIZE)
    val_gen = train_datagen.flow_from_directory(VAL_DIR, target_size=(IMAGE_HEIGHT, IMAGE_WIDTH), batch_size=BATCH_SIZE)
    
    # model = VGG19(input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3), classes = CLASSES)
    # # model = tf.keras.models.load_model(r'./VGGModel_0310.h5')
    
    # model.summary()
    # # optimizers.RMSprop(lr=LR)
    # model.compile(optimizer=optimizers.SGD(lr=LR), loss='categorical_crossentropy', metrics=['accuracy'])

    # history = model.fit_generator(train_gen, epochs=EPOCHS, validation_data=val_gen)
    
    model = Xception(include_top=False, weights='imagenet', input_tensor=None,input_shape=(IMAGE_HEIGHT,IMAGE_WIDTH,3))
    x = model.output
    x = Flatten()(x)
    
    # 增加 DropOut layer
    x = Dropout(0.2)(x)     #0.5               #########################
    
    # 增加 Dense layer，以 softmax 產生個類別的機率值
    output_layer = Dense(CLASSES, activation='softmax', name='softmax')(x)
    
    # 設定凍結與要進行訓練的網路層
    net_final = Model(inputs=model.input, outputs=output_layer)
    for layer in net_final.layers[:2]:
        layer.trainable = False
    for layer in net_final.layers[2:]:
        layer.trainable = True
    
    # 使用 Adam optimizer，以較低的 learning rate 進行 fine-tuning
    net_final.compile(optimizer=Adam(lr=LR), loss='categorical_crossentropy', metrics=['accuracy'])
    """
    model_path = os.path.join(MODEL_PATH, 'B802-43Q3-0606.h5')
    net_final = keras.models.load_model(model_path)
    
    model_summ = net_final.summary()
    print(model_summ)
    
    IMAGE_WIDTH = 175  # 50" 200    43" 175
    IMAGE_HEIGHT = 350
    LAYER_NAME = ""
    last_layer_name = 'block14_sepconv2_act'
    #model = tf.keras.applications.resnet.ResNet50(weights='imagenet', include_top=True)
    #grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_layer_name).output, model.output])
    image_path = r"D:\2021-0611-New Star\val-2type\Array\CACP3TH_BP_4706_1503_A.jpg"
    gcam_img0 = grad_cam(image_path, net_final, last_layer_name)
    
    #plt.imshow(gcam_img0)
    #plt.show()
    
    cv2.imwrite('craig_test.jpg', cv2.cvtColor(gcam_img0, cv2.COLOR_RGB2BGR))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    