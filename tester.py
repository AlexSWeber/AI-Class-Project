# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 22:42:48 2020

@author: Alex
"""
import pandas as pd 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import os
import csv

test_dir = 'C:/Users/Alex/Documents/School/AI/AI-Class-Project/test'
filenames = os.listdir(test_dir)
print(filenames)
df = pd.DataFrame({
    'filename': filenames
})

df.head(10) ##spot check 

IMAGE_WIDTH    =128
IMAGE_HEIGHT   =128
IMAGE_SIZE     =(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS =3
DROP_OUT_VALUE =0.25
FILTER_SIZE    =(3, 3)
POOL_SIZE      =(2, 2)

model = Sequential()

model.add(Conv2D(32, FILTER_SIZE, activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=POOL_SIZE))
model.add(Dropout(DROP_OUT_VALUE))

model.add(Conv2D(64, FILTER_SIZE, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=POOL_SIZE))
model.add(Dropout(DROP_OUT_VALUE))

model.add(Conv2D(128, FILTER_SIZE, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=POOL_SIZE))
model.add(Dropout(DROP_OUT_VALUE))


model.add(Conv2D(192, FILTER_SIZE, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=POOL_SIZE))
model.add(Dropout(0.35))

model.add(Conv2D(256, FILTER_SIZE, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=POOL_SIZE))
model.add(Dropout(0.45))


model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(DROP_OUT_VALUE))
model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()

model.load_weights("cat_dod_model.h5") 

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        color_mode="rgb",
        shuffle = False,
        class_mode='categorical',
        batch_size=1)

filenames = test_generator.filenames
nb_samples = len(filenames)

predict = model.predict_generator(test_generator,steps = nb_samples)

rows = predict.shape[0]
probs_list = []
for idx in range(rows):
    probs_list.append(predict[idx,1])

with open('submission.csv', 'w', newline='') as myfile:
     wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
     wr.writerow(['id','label'])
     for idx in range(rows):
         wr.writerow([idx+1, probs_list[idx]])