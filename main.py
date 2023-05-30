import tensorflow as tf
from tensorflow.keras import models,layers,datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

(X_train,y_train),(X_test,y_test)=datasets.cifar10.load_data() #veri setini tensorflow kütüphanesi aracılığı ile yüklüyoruz ve train ve test olarak ayırıyoruz 

X_train.shape #Train verisinin shapeni görüntülüyoruz

y_test=y_test.reshape(-1,) #verileri işleyebilmek için arraye boş bir satır daha eklememiz gerekir

resim_siniflari=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"] #kullanılacak sınıflandırmalar

def plot_sample(X,y,index): #İsteğe bağlı kontrol için daha sonradan fotoları görüntüleyebilmek için bu fonksiyonu kullanabiliriz 
  plt.figure(figsize=(15,2))
  plt.imshow(X[index])
  plt.xlabel(resim_siniflari[y[index]])

plot_sample(X_test, y_test, 0) #örnek


X_train=X_train/255 #RGB formatındaki fotoların değerlerini tek 0 ile 1 arasına indirgiyoruz 
X_test=X_test/255

deep_learnin_model=models.Sequential([
    layers.Conv2D(filters=512,kernel_size=(3,3),activation='relu',input_shape=(32,32,3)), #işlem kolaylığı sağlaması için Con2d ile fotoları 3x3 boyutlarda ölçeklendirerek veri setini daha işlenebilir hale getiriyoruz
    layers.MaxPooling2D((3,3)), #işlem ve hesaplama kolaylığı sağlamak için pooling uyguluyoruz

    layers.Conv2D(filters=256,kernel_size=(3,3),activation='relu'),
    layers.MaxPooling2D((3,3)),

    layers.Flatten(),                    #Burda girilen sinir ağı ve katman sayıları opsiyoneldir fakat bu sayılardan daha fazla olması sadece sistemin yavaşlamasına sebep olur herhangi bir artışa sebep olmaz 
    layers.Dense(256,activation='relu'), #Görsel işleme konusunda relu fonksiyonu daha işlevsel olduğu için relu kullanılmıştır
    layers.Dense(128,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax') #Sınıflandırma işlememiz 10 sınıftan oluştuğu için kesinlikle softmax ve 10 sinir ağından oluşmalı
])

deep_learnin_model.compile(optimizer='adam',                      #Görsel işleme için en uygun optimizer olan adam ile kayıp fonksiyonu için bidan fazla sınıflandırma olduğundan dolayı sparse kullanılmıştır
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])                  

deep_learnin_model.fit(X_train, y_train, epochs=25)               #Epoch sayısının 20 25 değerlerini üstünde olması veri setinin küçük olmasından kaynaklı olarak bir etki yaratmaz sadece ezbere sebeb olur 

deep_learnin_model.evaluate(X_test,y_test)

y_pred=deep_learnin_model.predict(X_test)
y_pred[:5]

y_classes =[np.argmax(element)for element in y_pred]              #Bu kısımlarda ise daha önce oluşturduğumuz görselleştirme fonksiyonu ile yapılan tahminler kontrol edilmiştir. 
y_classes[:5]

y_predictions_siniflari=[np.argmax(element)for element in y_pred]
y_predictions_siniflari[:5]

y_test[:5]

plot_sample(X_test,y_test,1348)

resim_siniflari[y_predictions_siniflari[1348]]
