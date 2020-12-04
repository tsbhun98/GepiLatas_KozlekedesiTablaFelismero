import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
import time

from PIL import Image
from sklearn.metrics import accuracy_score

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

from cleverhans.future.tf2.attacks import fast_gradient_method, projected_gradient_descent
from absl import app, flags

FLAGS = flags.FLAGS

#plottolo fuggveny
def plot_images(x, n):
    for i in range(n):
        plt.axis("off")
        plt.imshow(x[i])
        plt.show()
        p
data=[]
labels=[]

height = 30
width = 30
channels = 3
n_inputs = height * width*channels

#12 - Foutvonal
#13 - Elsobbsegadas kotelezo
#14 - STOP
#15 - Mindket iranybol behajtani tilos

for i in range(12, 16) :
    path = "c:/Users/tsbalazs/Gepi_latas/GTSRB/Final_Training/{}/".format(i)
    print(path)
    Class=os.listdir(path)
    for a in Class:
        try:
            image=cv2.imread(path+a)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            data.append(np.array(size_image))
            labels.append(i-12)
        except AttributeError:
            print(" ")
            
Cells=np.array(data)
labels=np.array(labels)

s=np.arange(Cells.shape[0])
np.random.seed(4)
np.random.shuffle(s)
Cells=Cells[s]
labels=labels[s]

# Tanitasi es validacios keszlet felosztasa (80-20)
(X_train,X_val)=Cells[(int)(0.2*len(labels)):],Cells[:(int)(0.2*len(labels))]
X_train = X_train.astype('float32')/255 
X_val = X_val.astype('float32')/255
(y_train,y_val)=labels[(int)(0.2*len(labels)):],labels[:(int)(0.2*len(labels))]

#one hote encoding hasznalata
y_train = to_categorical(y_train, 4)
y_val = to_categorical(y_val, 4)

#https://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/
input_shape = X_train.shape[1:] # (30,30,3)

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#modell tanitasa
epochs = 2

start = time.time()

history = model.fit(X_train, y_train, batch_size=32, epochs=epochs, validation_data=(X_val, y_val))

end = time.time()

print("Modell tanítási ideje: {:.4f} s".format(end-start))

#tesztkeszlet beolvasasa
test = pd.read_csv('c:/Users/tsbalazs/Gepi_latas/GTSRB/GT-final_test.csv', sep=';')

X_test = []
y_test = []
for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
    if class_id>=12 and class_id<=15:
        path = os.path.join("c:/Users/tsbalazs/Gepi_latas/GTSRB/Final_Test/", file_name)
        try:
            image=cv2.imread(path)
            image_from_array = Image.fromarray(image, 'RGB')
            size_image = image_from_array.resize((height, width))
            X_test.append(np.array(size_image))
            y_test.append(class_id-12)
        except AttributeError:
            print(" ")

y_test = np.array(y_test)        
X_test = np.array(X_test)
X_test = X_test.astype('float32')/255 

#Pontossag a tesztkeszleten
pred = model.predict_classes(X_test)
acc = accuracy_score(y_test, pred)
print("Pontosság a tesztkészleten: {:.4f} %".format(acc*100))

#Ellenseges peldakon pontossag tesztelese
#Forras:https://github.com/tensorflow/cleverhans/blob/master/tutorials/future/tf2/mnist_tutorial.py (2020.11.26)

#ellenseges peldak generalasa az FGSM modszerrel
start = time.time()

x_fgm = fast_gradient_method(model, X_test, 0.2, np.inf)

end = time.time()

print("Ellenséges példák generálási ideje az FGSM módszerrel: {:.4f} s".format(end-start))

test_acc_fgsm = tf.metrics.SparseCategoricalAccuracy()

y_pred_fgm = model(x_fgm)
test_acc_fgsm(y_test, y_pred_fgm)
print('Pontosság az FGSM által generált képeken: {:.4f} %'.format(test_acc_fgsm.result() * 100))

#ellenseges peldak generalasa a PGD modszerrel
start = time.time()

x_pgd = projected_gradient_descent(model, X_test, 0.2, 0.01, 40, np.inf)

end = time.time()

print("Ellenséges példák generálási ideje a PGD módszerrel: {:.4f} s".format(end-start))

test_acc_pgd = tf.metrics.SparseCategoricalAccuracy()

y_pred_pgd = model(x_pgd)
test_acc_pgd(y_test, y_pred_pgd)
print('Pontosság a PGD által generált képeken: {:.4f} %'.format(test_acc_pgd.result() * 100))