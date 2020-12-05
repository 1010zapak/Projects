from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
mainDIR = os.listdir('C:\\Users\\anubh\\Downloads\\chest_xray\\chest_xray')
print(mainDIR)

train_folder= 'C:\\Users\\anubh\\Downloads\\chest_xray\\chest_xray\\train\\'
val_folder = 'C:\\Users\\anubh\\Downloads\\chest_xray\\chest_xray\\val\\'
test_folder = 'C:\\Users\\anubh\\Downloads\\chest_xray\\chest_xray\\test\\'

# os.listdir(train_folder)
# train_n = train_folder+'NORMAL\\'
# train_p = train_folder+'PNEUMONIA\\'
# print(train_n)
# print(train_p)

train = ImageDataGenerator(rescale = 1./255.,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
validation = ImageDataGenerator(rescale=1./255.)
test = ImageDataGenerator(rescale=1./255.)


train_set = train.flow_from_directory(train_folder,target_size=(256,256),class_mode='binary',batch_size=32)
val_set = validation.flow_from_directory(val_folder,target_size=(256,256),batch_size=32,class_mode='binary')
test_set = test.flow_from_directory(test_folder,target_size = (256,256),batch_size = 32,class_mode = 'binary')
# epoch badhane me overfit ho raha tha

model = tf.keras.Sequential(
	[
	tf.keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(256,256,3)),
	tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

	tf.keras.layers.Flatten(),

	tf.keras.layers.Dense(1024,activation='relu'),
	tf.keras.layers.Dense(1,activation='sigmoid')]
)
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
hist = model.fit(train_set,epochs=15,validation_data=val_set,steps_per_epoch=163,validation_steps=624)
loss,acc = model.evaluate(test_set)
print('The testing loss is :',loss)
print('The testing accuracy is :',acc*100,'%')


from keras.preprocessing import image

img_width, img_height = 256,256
img = image.load_img('C:\\Users\\anubh\\Desktop\\CXRAy\\NORMAL\\IM-0151-0001.jpeg', target_size = (img_width, img_height,3))
img = image.img_to_array(img)
img = np.expand_dims(img, axis = 0)

img2 = image.load_img('C:\\Users\\anubh\\Downloads\\chest_xray\\chest_xray\\train\\PNEUMONIA\\person8_bacteria_37.jpeg', target_size = (img_width, img_height,3))
img2 = image.img_to_array(img2)
img2 = np.expand_dims(img2, axis = 0)

img3 = image.load_img('C:\\Users\\anubh\\Downloads\\chest_xray\\chest_xray\\train\\PNEUMONIA\\person28_bacteria_139.jpeg', target_size = (img_width, img_height,3))
img3 = image.img_to_array(img3)
img3 = np.expand_dims(img3, axis = 0)

print(model.predict(img))
print(model.predict(img2))
print(model.predict(img3))
print("Done")

# Accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training set', 'Validation set'], loc='upper left')
plt.show()

