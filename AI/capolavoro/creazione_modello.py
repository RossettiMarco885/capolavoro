# Importazione librerie
from keras.datasets import mnist
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

import matplotlib.pyplot as plt  # plotting library


from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam, RMSprop
from keras import backend as K


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


# Caricamento del dataset MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizzazione dei dati
x_train, x_test = x_train / 255.0, x_test / 255.0

# Conversione etichette in one-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Visualizzazione di alcuni esempi
plt.subplot(121)
plt.imshow(x_train[0], cmap="gray")
plt.title("Esempio training")
plt.subplot(122)
plt.imshow(x_test[0], cmap="gray")
plt.title("Esempio test")
plt.show()

# Definizione dell'architettura della CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(10, activation="softmax"))

# Compilazione del modello
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
)

# Definizione callback per early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5)

# Addestramento del modello
model.fit(
    x_train,
    y_train,
    epochs=20,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping],
)

# Valutazione del modello
model.evaluate(x_test, y_test)

# Salvataggio del modello
model.save("C:\\Users\\rosse\Documents\\GitHub\\capolavoro\\AI\\capolavoro\\mnist_9.keras")
print("modello salvato")
