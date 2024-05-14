import streamlit as st
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import keras as ke
from PIL import Image
from tensorflow.keras import datasets,layers,models
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# (trainImages,trainLabels),(testImages,testLabels) = datasets.cifar10.load_data()
# trainImages,testImages = trainImages/255,testImages/255

ClassNames = ['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Truck']

# for i in range(16):
#     plt.subplot(4,4,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(trainImages[i],cmap=plt.cm.binary)
#     plt.xlabel(ClassNames[trainLabels[i][0]])
    
# plt.show()    

# trainImages = trainImages[:20000]
# trainLabels = trainLabels[:20000]
# testImages = testImages[:4000]
# testLabels = testLabels[:4000]

# model = models.Sequential()
# model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64,(3,3),activation='relu'))
# model.add(layers.MaxPooling2D((2,2)))
# model.add(layers.Conv2D(64,(3,3),activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64,activation='relu'))
# model.add(layers.Dense(10,activation='softmax'))

# model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# model.fit(trainImages,trainLabels,batch_size = 1,epochs=10,validation_data=(testImages,testLabels))

# loss, accuracy = model.evaluate(trainImages,trainLabels)
# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")

# model.save('ImageClassifier_V1.0.keras')

# model = models.load_model('ImageClassifier_V1.0.model')
# model = ke.layers.TFSMLayer('ImageClassifier_V1.0.model', call_endpoint='serving_default')
model = models.load_model('ImageClassifier_V1.0.keras')
st.title("Image Classification trained on Cifar-10")



img = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
display_image = None
if img is None:
    img = cv.imread('testimg1.jpg')
    display_image = img 
else:
    img = np.array(Image.open(img))
    display_image = img 
y=40
x=50
h=img.shape[0]-40
w=img.shape[1]-50
img = img[y:h,x:w]
img = cv.resize(img, (32, 32), interpolation = cv.INTER_LINEAR)
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

# plt.imshow(img,cmap=plt.cm.binary)
st.image(display_image, caption=f"Processed image", use_column_width=True,)

prediction = model.predict(np.array([img])/255)
index = np.argmax(prediction)
print(f"Pridiction is {ClassNames[index]}")
# plt.xlabel(ClassNames[index])
st.write(ClassNames[index])

# plt.show()