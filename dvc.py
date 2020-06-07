# %% [markdown]
# # In this kernel we’ll try to identify correctly if the image we’re seeing is a dog=1 or a cat=0 .
# # In order to do so I’ll use dog & cat image dataset as input to a data augmentation generator + CNN . 
# # After optimzation process - optimal layers are 5 + 2 FC layers 
# # I also run semi grid search on dropout , filter size .etc.. 
# 

# %% [markdown]
# # Imports

# %% [code]
import numpy as np
import pandas as pd 
import zipfile
from keras.preprocessing.image import ImageDataGenerator, load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization
import os
#print(check_output(["ls", "../input/"]).decode("utf8"))

# %% [markdown]
# # Extract all train images

# %% [code]
with zipfile.ZipFile('C:/Users/Alex/Documents/School/AI/train.zip', 'r') as z:    
    z.extractall(".") 
#print(check_output(["ls", "train"]).decode("utf8"))
   

# %% [markdown]
# # Get  filenames paths + define category (1 - dog , 0 - cat) and fill both of those varibles inside dataframe

# %% [code]
filenames = os.listdir("train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

df.head(10) ##spot check 


# %% [markdown]
# # Define constant for CNN & data augmentation generator

# %% [code]
FAST_RUN       = True
#FAST_RUN_EPOCHS=3
FAST_RUN_EPOCHS=2
IMAGE_WIDTH    =128
IMAGE_HEIGHT   =128
IMAGE_SIZE     =(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS =3
DROP_OUT_VALUE =0.25
FILTER_SIZE    =(3, 3)
POOL_SIZE      =(2, 2)

# %% [markdown]
# # Load & plot random image from DF

# %% [code]
sample = random.choice(filenames)
image = load_img("train/"+sample)
plt.imshow(image)

# %% [markdown]
# # Count  dog & cat images

# %% [code]
df['category'].value_counts().plot.bar()

# %% [markdown]
# # Inside training data we have 24000 dog & cat images

# %% [markdown]
# # We have 6 tiers where inside each tier we have : 
# # Conv layer ->  batch normalization -> max pooling -> dropout 
# # Last layer will include FC layer : flatten ->dense ->droput ->dense 
# # Input layer: We'll take the images and resample image from Height X Weight X 3 ->  256 X256 X 3
# # Output Layer: classify dog=1 or cat =0 

# %% [code]


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

# %% [markdown]
# #  Early Stop - prevent over fitting we will stop the learning after 10 epochs or val_loss not decreased 

# %% [code]
earlystop = EarlyStopping(patience=15)

# %% [markdown]
# # Learning Rate Reduction-reduce learning rate when then accuracy not increase for 2 steps

# %% [code]
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.4, 
                                            min_lr=0.0001)

# %% [code]
callbacks = [earlystop, learning_rate_reduction]

# %% [markdown]
# Because we will use image genaretor `with class_mode="categorical"`. We need to convert column category into string. Then imagenerator will convert it one-hot encoding which is good for our classification. 
# 
# So we will convert 1 to dog and 0 to cat

# %% [code]
df.head()

# %% [code]
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'}) 
df.head()

# %% [code]
train_df, validate_df = train_test_split(df, test_size=0.25, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

# %% [code]
train_df['category'].value_counts().plot.bar()

# %% [code]
validate_df['category'].value_counts().plot.bar()

# %% [code]
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15
total_train

# %% [markdown]
# # Traning Generator

# %% [code]
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

# %% [markdown]
# ### Validation Generator

# %% [code]
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)

# %% [markdown]
# # See how our generator work

# %% [code]
example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)

# %% [code]
plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()

# %% [markdown]
# # Fit Model

# %% [code]
epochs=FAST_RUN_EPOCHS if FAST_RUN else 16
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)

# %% [markdown]
# > # Save Model

# %% [code]
model.save_weights("cat_dod_model.h5")    

# %% [markdown]
# # Plot training images with dog/cat output

# %% [code]
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()

# %% [markdown]
# huge thanks to Uysim Keras CNN Dog or Cat Classification notebook 