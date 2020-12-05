# example of using ImageDataGenerator to normalize images
from keras.datasets import mnist
from keras.utils import to_categorical

#from keras.layers import Conv2D
#from keras.layers import MaxPooling2D
#from keras.layers import Dense
#from keras.layers import Flatten

#importing from keras gives error import from tensorflow due to gpu issues

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator
import cv2
(trainX, trainY), (testX, testY) = mnist.load_data()

width, height, channels = trainX.shape[1], trainX.shape[2], 1
trainX = trainX.reshape((trainX.shape[0], width, height, channels))
testX = testX.reshape((testX.shape[0], width, height, channels))

trainY = to_categorical(trainY)
testY = to_categorical(testY)

print('Train min=%.3f, max=%.3f' % (trainX.min(), trainX.max()))
print('Test min=%.3f, max=%.3f' % (testX.min(), testX.max()))

datagen = ImageDataGenerator(rescale=1.0/255.0)

train_iterator = datagen.flow(trainX, trainY, batch_size=64)
test_iterator = datagen.flow(testX, testY, batch_size=64)
print('Batches train=%d, test=%d' % (len(train_iterator), len(test_iterator)))

batchX, batchy = train_iterator.next()
print('Batch shape=%s, min=%.3f, max=%.3f' % (batchX.shape, batchX.min(), batchX.max()))

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(height,width,channels)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(train_iterator, steps_per_epoch=len(train_iterator), epochs=5)

_, acc = model.evaluate(test_iterator, steps=len(test_iterator), verbose=0)
print('Test Accuracy: %.3f' % (acc * 100))
print("Image shape",height,width,channels)
# a = cv2.imread("C:\\Users\\anubh\\Desktop\\ample_image.png", cv2.IMREAD_GRAYSCALE)
# print(model.predict(a))