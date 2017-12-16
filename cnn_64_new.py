import sklearn
import tensorflow as tf
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import xml.dom.minidom
import os.path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
from sklearn.model_selection import train_test_split
from scipy.misc import imresize 
import keras
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn import linear_model, decomposition
import itertools 
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
import random
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.initializers import RandomNormal  
from keras.layers.normalization import BatchNormalization

#############################################################
# region of segmentation
def img_dim(obj):
    x_max = int(math.ceil(float(obj.find('bndbox').find('xmax').text)))
    y_max = int(math.ceil(float(obj.find('bndbox').find('ymax').text)))
    x_min = int(math.ceil(float(obj.find('bndbox').find('xmin').text)))
    y_min = int(math.ceil(float(obj.find('bndbox').find('ymin').text)))
    return [x_max,y_max,x_min,y_min]


            
# convert name to label
def name2label(name):
    name_list = ['dog',
 'sheep',
 'person',
 'pottedplant',
 'boat',
 'bird',
 'horse',
 'bus',
 'tvmonitor',
 'sofa',
 'diningtable',
 'car',
 'motorbike',
 'train',
 'bicycle',
 'cat',
 'chair',
 'cow',
 'bottle',
 'aeroplane']
    
    return name_list.index(name)


# save seg data
def all_image_data_label(file, siz):
    image_buffer  =  []
    label_buffer = []
    for i in range(len(file)):
        tree = ET.parse('/home/cc/notebook/final_7866/VOCdevkit/VOC2012/Annotations/'+file[i])
        root = tree.getroot()
        img_path = '/home/cc/notebook/final_7866/VOCdevkit/VOC2012/JPEGImages/'+root.find('filename').text
        img = mpimg.imread(img_path)
        image_buffer.append(imresize(img, siz)/255) # resize and 0 to 1
        obj = root.findall('object')
        label_img = [0]*20
        for j in range(len(obj)):
            name = obj[j].find('name').text
            label_img[name2label(name)] = 1
    
        label_buffer.append(label_img)
    
    return image_buffer,label_buffer 

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def modelAnalysis(test_y, predict,target_na,cna):        
    print(classification_report(test_y, predict, target_names=target_na))
    print('accuracy', accuracy_score(test_y, predict))
    mat = confusion_matrix(test_y, predict)
    
    target_na = np.array(target_na)
    plt.figure(figsize=(10,10))
    a = plot_confusion_matrix(mat, classes = target_na, normalize = False)
    plt.grid(b=False)
    plt.savefig(cna)
    #plt.show()
    

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    temp = []
    mat = cm
    for i in range(len(mat)):
        temp.append((mat[i, :].astype(np.float64)/(sum(mat[i])+1e-32).tolist()))
    
    mat = np.array(temp)
    
    plt.imshow(mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        thresh = sum(cm[i])/ 2.
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    

def first_k(output,kn):
    for i in range(kn):
        output[np.arange(len(output)),np.argmax(output,axis = 1)] = -1

    output[np.where(output>0)] = 0
    output[np.where(output<-0.5)] = 1
    return output.astype(np.int)

def convert_label(y_train,predict_score,kn):
    predict_y = first_k(predict_score,kn)
    tmp = y_train&predict_y
    label = []
    pred = []
    for i in range(len(tmp)):
        if np.where(tmp[i]==1)[0].shape[0]<0.1: # no label match  
            label.append(random.choice(np.where(y_train[i]>0.7)[0])) # train label
            pred.append(np.argmax(predict_score[i]))
        else:
            tmp_label = random.choice(np.where(tmp[i]>0.7)[0])
            pred.append(tmp_label)
            label.append(tmp_label)
            
    return label, pred



#######################################################################
file = os.listdir('VOCdevkit/VOC2012/Annotations')
img_size = 64
kn = 1

name_list = ['dog',
 'sheep',
 'person',
 'pottedplant',
 'boat',
 'bird',
 'horse',
 'bus',
 'tvmonitor',
 'sofa',
 'diningtable',
 'car',
 'motorbike',
 'train',
 'bicycle',
 'cat',
 'chair',
 'cow',
 'bottle',
 'aeroplane']


###########################################################################
file = os.listdir('VOCdevkit/VOC2012/Annotations')
dummy_label = np.random.randint(0,10,len(file))

#train_file, test_file, dummy_train_y, dummy_test_y = train_test_split(file, dummy_label, test_size=0.3, random_state=99999)

train_file = pd.read_csv('train_index.csv').values[:,0].tolist()
test_file = pd.read_csv('test_index.csv').values[:,0].tolist()

img, label = all_image_data_label(train_file, (img_size,img_size)) # resize 128 to 128
test_img, test_label = all_image_data_label(test_file, (img_size,img_size)) # resize 128 to 128

x_train = np.array(img)
y_train = label
x_test = np.array(test_img)
y_test = test_label
#y_train = keras.utils.to_categorical(y_train, len(name_list))
#y_test = keras.utils.to_categorical(y_test, len(name_list))
y_train = np.array(y_train)
y_test = np.array(y_test)

num_classes = 20
num_predictions = 20
batch_size    = 128
epochs        = 100

model = Sequential()

model.add(Conv2D(124, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.0002), kernel_initializer=RandomNormal(stddev = 0.01),input_shape=x_train.shape[1:]))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(96, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.0002), kernel_initializer=RandomNormal(stddev = 0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.0002), kernel_initializer=RandomNormal(stddev = 0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding = 'same'))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.0002), kernel_initializer=RandomNormal(stddev = 0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=keras.regularizers.l2(0.0002), kernel_initializer=RandomNormal(stddev = 0.01)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), padding = 'same'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(num_classes))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

datagen = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,horizontal_flip=True)  

datagen.fit(x_train)
    
model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size),steps_per_epoch=int(np.ceil(x_train.shape[0]/float(batch_size))),epochs=epochs,validation_data=(x_test, y_test),workers=16)

model.save_weights('final_weight.h5')
