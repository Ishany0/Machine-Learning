import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist

# Load pre-trained models without top layer
resnet_base = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

xception_base = tf.keras.applications.Xception(
    include_top=False, 
    weights='imagenet',
    input_shape=(224, 224, 3)
)

# Freeze base model layers
for layer in resnet_base.layers:
    layer.trainable = False
for layer in xception_base.layers:
    layer.trainable = False
(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

train_images=train_images/255.0
test_images=test_images/255.0

number_of_classes=train_labels.max()+1

weight=tf.keras.ResNet50(weights=None,input_shape=(28,28,1),classes=number_of_classes)

model=tf.keras.Sequential(
    tf.keras.layers.weight(include_top=False,input_shape=(28,28,1)
))

model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),metrics=['accuracy'])
# Reshape images to add channel dimension
train_images = train_images.reshape(-1, 28, 28, 1)
test_images = test_images.reshape(-1, 28, 28, 1)

# Convert grayscale to RGB by repeating channels
train_images = np.repeat(train_images, 3, axis=3)
test_images = np.repeat(test_images, 3, axis=3)

# Resize images to match ResNet50 and Xception input requirements
train_images_resized = tf.image.resize(train_images, (224, 224))
test_images_resized = tf.image.resize(test_images, (224, 224))

# Create ResNet50 and Xception models with pretrained weights
resnet_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
xception_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the pre-trained layers
for layer in resnet_model.layers:
    layer.trainable = False
for layer in xception_model.layers:
    layer.trainable = False

# Add custom classification layers
model = tf.keras.Sequential([
    resnet_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(number_of_classes, activation='softmax')
])
history=model.fit(train_images,train_labels,epochs=5,verbose=True,validation_data=(test_images,test_labels))

y_pred_ann = np.argmax(model.predict(test_images), axis=1)
print("\nANN Classification Report:\n")
print(classification_report(test_labels,y_pred_ann,digits=4))

x_train=train_images.reshape(len(train_images),-1)
x_test=test_images.reshape(len(test_images),-1)

log_reg=LogisticRegression(max_iter=1000)
log_reg.fit(x_train,train_labels)
y_pred_lr=log_reg.predict(x_test)
print('\nLogistic Regression Report:\n')
print(classification_report(test_labels,y_pred_lr,digits=4))

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)
svm_clf=SVC(kernel='linear')
svm_clf.fit(x_train_scaled[:10000],train_labels[:10000])
y_pred_svm=svm_clf.predict(x_test_scaled)
print('\nSVM Report:\n')
print(classification_report(test_labels,y_pred_svm,digits=4))


plt.imshow(test_images[33],cmap='gray')
plt.title(f'True Label: {test_labels[33]}')
plt.show()

'''
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
'''
