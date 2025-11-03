#basic ml model using neural network and tensorflow

import cv2 as cv        #open cv library lets us add our own images and do image processing
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf          #tensorflow library for building neural networks


#keras is used for building neural networks

mnist=tf.keras.datasets.mnist        #mnist dataset of handwritten digits
(x_train,y_train),(x_test,y_test)=mnist.load_data()    #it already splits data into training and testing sets

x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)      #we need to scale x data because pixel values range from 0-255
# y data is already in 0-9 range so no need to scale

model = tf.keras.models.Sequential()  #and ordinary feedforward neural network
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))  #flatten layer to convert 2D images to 1D array (like a grid here)
model.add(tf.keras.layers.Dense(units=128,activation='relu'))  #hidden layer with 128 neurons and relu activation function( -1 for neg and 0 for pov)
model.add(tf.keras.layers.Dense(units=128,activation='relu')) #2 hidden layers now
model.add(tf.keras.layers.Dense(units=10,activation='softmax')) #output layer 
#softmax scales the values down so that they all add up to 1 so that we can get the probability of the result

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy']) #adam=adaptive moment estimation, Works well out-of-the-box for many problems

model.fit(x_train,y_train,epochs=3) #epochs mean how many times we are going to use the same data over and over again

loss,accuracy = model.evaluate(x_test,y_test)
print(accuracy)
print(loss)


for x in range(1,4):
    img=cv.imread(f'{x}.png')[:,:,0]
    img=np.invert(np.array([img]))
    prediction=model.predict(img)
    print(f'The result is probably:{np.argmax(prediction)}')    #argmax gives us the index of the highest prediction
    plt.imshow(img[0],cmap=plt.cm.binary)
    plt.show()