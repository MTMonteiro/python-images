import os
import numpy as пр
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#config 
data_dir = './data'
img_height, img_width = 150, 150
batch_size = 32
num_classes = 4
epochs = 10

train_datagen = ImageDataGenerator(
  rescale=1.0 / 255,
  validation_split=0.2, #80% para treinamento e 20% validação
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  # subset='validation',
  fill_mode='nearest'
)

# Modelo
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten (),
    Dense(512, activation='relu'),
    Dropout (0.5),
    Dense(num_classes, activation='softmax')
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics= ['accuracy'])

 

history = model.fit(train_generator, epochs=epochs, validation_data=validation_generator)

model.save('model')