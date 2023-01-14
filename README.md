# Data-Science-with-ALI
#CNN (Autoencoder/DLSS) / low resolution to high resolution/ blur images to clear images

from google.colab import drive
drive.mount('/content/drive/')

import tensorflow
tensorflow.test.is_gpu_available()

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array 
import cv2



#now we will make input images by lowering resolution without changing the size
import cv2
def pixalate_image(image, scale_percent = 5):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    small_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
  
     # scale back to original size
    width = int(small_image.shape[1] * 100 / scale_percent)
    height = int(small_image.shape[0] * 100 / scale_percent)
    dim = (width, height)
    low_res_image = cv2.resize(small_image, dim, interpolation =  cv2.INTER_AREA)
    return low_res_image
    
    
high_res_training = "/content/drive/MyDrive/high_res_training"
low_res_training = "/content/drive/MyDrive/low_res_training"
high_res_testing = "/content/drive/MyDrive/high_res_testing"
low_res_testing="/content/drive/MyDrive/low_res_testing"




for file in os.listdir(high_res_training):
    print(os.path.join(high_res_training,file))
    img = cv2.imread(os.path.join(high_res_training,file))
    print(img.shape)
    cv2.imwrite(os.path.join(low_res_training,file),pixalate_image(img))
    
    
    
    
    high_res_img_training =[]
for filename in os.listdir(high_res_training):
    if filename.endswith(".jpg"):
        img = image.load_img(os.path.join(os.getcwd(), os.path.join(high_res_training, filename)), target_size=(256,256))
        high_res_img_training.append(image.img_to_array(img))
high_res_img_training= np.array(high_res_img_training)
high_res_img_training



low_res_img_training =[]
for filename in os.listdir(low_res_training):
    if filename.endswith(".jpg"):
        img = image.load_img(os.path.join(os.getcwd(), os.path.join(low_res_training, filename)), target_size=(256,256))
        low_res_img_training.append(image.img_to_array(img))
low_res_img_training= np.array(high_res_img_training)
low_res_img_training


print("high_res_img_training",high_res_img_training.shape)

print("low_res_img_training",low_res_img_training.shape)


for file in os.listdir(high_res_testing):
    print(os.path.join(high_res_testing,file))
    img = cv2.imread(os.path.join(high_res_testing,file))
    print(img.shape)
    cv2.imwrite(os.path.join(low_res_testing,file),pixalate_image(img))
    
    
high_res_img_testing =[]
for filename in os.listdir(high_res_testing):
    if filename.endswith(".jpg"):
        img = image.load_img(os.path.join(os.getcwd(), os.path.join(high_res_testing, filename)), target_size=(256,256))
        high_res_img_testing.append(image.img_to_array(img))
high_res_img_testing= np.array(high_res_img_testing)
high_res_img_testing


low_res_img_testing =[]
for filename in os.listdir(low_res_testing):
    if filename.endswith(".jpg"):
        img = image.load_img(os.path.join(os.getcwd(), os.path.join(low_res_testing, filename)), target_size=(256,256))
        low_res_img_testing.append(image.img_to_array(img))
low_res_img_testing= np.array(low_res_img_testing)
low_res_img_testing


print("low_res_img_testing",low_res_img_testing.shape)


def show_test_data(X,n=5,title=""):
    plt.figure(figsize=(30,40))
    for i in range(n):
        ax=plt.subplot(3,n,i+1)
        plt.imshow(image.array_to_img(X[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle(title,fontsize=15)
    
    
show_test_data(high_res_img_testing,title="high res img")
show_test_data(low_res_img_testing,title="low_res")




#CNN
input_layer = Input(shape=(256, 256, 3))  
#encoding architecture 
x1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(input_layer)
x2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x1)
x3 = MaxPool2D(padding='same')(x2)
x4 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x3)
x5 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x4)
x6 = MaxPool2D(padding='same')(x5)
encoded = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x6)
#encoded = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
# decoding architecture
x7 = UpSampling2D()(encoded)
x8 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x7)
x9 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x8)
x10 = Add()([x5, x9])
x11 = UpSampling2D()(x10)
x12 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x11)
x13 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x12)
x14 = Add()([x2, x13])
# x3 = UpSampling2D((2, 2))(x3)
# x2 = Conv2D(128, (3, 3), activation='relu', padding='same')(x3)
# x1 = Conv2D(256, (3, 3), activation='relu', padding='same')(x2)
output_layer = Conv2D(3, (3, 3), padding='same',activation='relu', kernel_regularizer=regularizers.l1(10e-10))(x14)
#autoencoder = Model(Input_img, decoded)
#autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])



autoencoder_AE = Model(input_layer,output_layer)
autoencoder_AE.compile(optimizer='Adam', loss='mse',metrics=["accuracy"])
autoencoder_AE.summary()

#training
history= autoencoder_AE.fit(low_res_img_training,high_res_img_training,
                epochs=500,
                batch_size=20,
                shuffle=True,
                validation_data=(low_res_img_testing,high_res_img_testing))
                

 
#visualization
from matplotlib import pyplot as plt
import numpy as np
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = list(np.arange(0,50, 1))
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


loss_train = history.history['accuracy']
loss_val = history.history['val_accuracy']
epochs = list(np.arange(0,50, 1))
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
    

autoencoder_AE.summary()


results =autoencoder_AE.evaluate(low_res_img_training,high_res_img_training)
print('train_loss,train_accuracy',results)

results =autoencoder_AE.evaluate(low_res_img_testing,high_res_img_testing)
print('val_loss,val_accuracy', results)
                
# note that we take them from the *test* set
encoded_imgs = autoencoder_AE.predict(low_res_img_testing)
decoded_imgs = autoencoder_AE.predict(encoded_imgs) 


def show_data(X,n=5,title=""):
    plt.figure(figsize=(20,30))
    for i in range(n):
        ax=plt.subplot(3,n,i+1)
        plt.imshow(image.array_to_img(X[i]))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle(title,fontsize=15)
    
    
    
from tensorflow.keras.utils import load_img, img_to_array 
import keras.utils as image


show_test_data(high_res_img_testing,title="original test image with good resolution ")
show_data(low_res_img_testing,title="low resolution encoded img")
show_data(decoded_imgs,title="decoded image,reconstructed img")



