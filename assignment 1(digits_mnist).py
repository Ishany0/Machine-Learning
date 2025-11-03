import tensorflow as tf
import matplotlib.pyplot as plt
mnist=tf.keras.datasets.mnist
(train_images,train_labels),(valid_images,valid_labels)=mnist.load_data()
number_of_classes=train_labels.max()+1
model=tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(number_of_classes)
])
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
history=model.fit(train_images,train_labels,epochs=5,verbose=True,validation_data=(valid_images,valid_labels))
prediction=model.predict(train_images[0:10])
print(prediction)
data_idx=33
plt.figure()
plt.imshow(train_images[data_idx],cmap='gray')
plt.imshow(valid_images[data_idx],cmap='gray')
plt.colorbar()
plt.grid(False)
plt.show()

x_values=range(number_of_classes)
plt.figure()
plt.bar(x_values,model.predict(train_images[data_idx:data_idx+1]).flatten())
plt.xticks(range(10))
plt.show()

