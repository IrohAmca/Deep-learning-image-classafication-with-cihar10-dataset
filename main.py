import tensorflow as tf
from tensorflow.keras import models,layers,datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

(X_train,y_train),(X_test,y_test)=datasets.cifar10.load_data()

X_train.shape

y_test=y_test.reshape(-1,)

resim_siniflari=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

def plot_sample(X,y,index):
  plt.figure(figsize=(15,2))
  plt.imshow(X[index])
  plt.xlabel(resim_siniflari[y[index]])

plot_sample(X_test, y_test, 0)


X_train=X_train/255
X_test=X_test/255

deep_learnin_model=models.Sequential([
    layers.Conv2D(filters=512,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)),
    layers.MaxPooling2D((3,3)),

    layers.Conv2D(filters=256,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((3,3)),

    layers.Flatten(),
    layers.Dense(256,activation='relu'),
    layers.Dense(128,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])

deep_learnin_model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

deep_learnin_model.fit(X_train, y_train, epochs=40)

deep_learnin_model.evaluate(X_test,y_test)

y_pred=deep_learnin_model.predict(X_test)
y_pred[:5]

y_classes =[np.argmax(element)for element in y_pred]
y_classes[:5]

y_predictions_siniflari=[np.argmax(element)for element in y_pred]
y_predictions_siniflari[:5]

y_test[:5]

plot_sample(X_test,y_test,1348)

resim_siniflari[y_predictions_siniflari[1348]]