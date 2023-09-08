import tensorflow as tf

# Importar o conjunto de dados
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Redimensionar as imagens
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Normalizar as imagens
x_train = x_train / 255.0
x_test = x_test / 255.0

# Construir a rede neural
model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar o modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Treinar o modelo
model.fit(x_train, y_train, epochs=10)

# Avaliar o modelo
model.evaluate(x_test, y_test)
model.save('model')