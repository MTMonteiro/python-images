import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import scipy
from scipy import optimize  # Para funções de otimização
from scipy import stats     # Para funções estatísticas


model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(4, activation='softmax')  # 10 classes para números de 0 a 9
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'],
              run_eagerly=True)


# -------------------------------------------------


train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(28, 28),
    batch_size=32,
    class_mode='sparse',
    color_mode='grayscale')

validation_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(28, 28),
    batch_size=32,
    class_mode='sparse',
    color_mode='grayscale')


# treinamento 
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32)

# avaliação 
test_loss, test_acc = model.evaluate(validation_generator, steps=validation_generator.samples // 32)
print(f'Acurácia no conjunto de teste: {test_acc * 100:.2f}%')

# Salve o modelo treinado em um arquivo .h5
model.save('modelo_de_reconhecimento.h5')