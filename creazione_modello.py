import os
import numpy as np
from keras.layers import Input # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from keras.datasets import mnist # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import tensorflow as tf

# Percorso alla tua cartella con le immagini dei numeri scritti a mano
handwritten_numbers_dir = "C:\\Users\\rosse\\Documents\\GitHub\\capolavoro\\IMG_ADDESTRAMENTO\\"

# Liste per memorizzare immagini ed etichette
images = []
labels = []

# Scansione dei file nella cartella
for filename in os.listdir(handwritten_numbers_dir):
    if filename.endswith(".png"):
        # Carica l'immagine e convertila in array numpy
        img = image.load_img(os.path.join(handwritten_numbers_dir, filename), target_size=(28, 28), color_mode="grayscale")
        img_array = image.img_to_array(img)
        # Normalizza l'array dell'immagine
        img_array /= 255.0
        # Aggiungi l'array dell'immagine alla lista delle immagini
        images.append(img_array)
        # Estrai l'etichetta dal nome del file (il primo carattere dell'immagine è il numero disegnato)
        label = int(filename[0])
        labels.append(label)

# Converte le liste in array numpy
x_tuoi = np.array(images)
y_tuoi = np.array(labels)

# Espansione delle dimensioni delle immagini del tuo dataset
x_tuoi = np.expand_dims(x_tuoi, axis=-1)

# Numero totale di classi (da 0 a 9)
num_classes = 10

# Converti le etichette del tuo dataset in one-hot encoding
y_tuoi = to_categorical(y_tuoi, num_classes=num_classes)

# Suddivisione dei tuoi dati in set di addestramento e di test
x_tuoi_train, x_tuoi_test, y_tuoi_train, y_tuoi_test = train_test_split(
    x_tuoi, y_tuoi, test_size=0.2, random_state=42
)

# Rimuovi la dimensione aggiuntiva da x_tuoi_train e x_tuoi_test
x_tuoi_train = np.squeeze(x_tuoi_train, axis=-1)
x_tuoi_test = np.squeeze(x_tuoi_test, axis=-1)

# Caricamento del dataset MNIST
(x_mnist_train, y_mnist_train), (x_mnist_test, y_mnist_test) = mnist.load_data()

# Espansione delle dimensioni delle immagini di MNIST per farle corrispondere a quelle del tuo dataset
x_mnist_train = np.expand_dims(x_mnist_train, axis=-1)
x_mnist_test = np.expand_dims(x_mnist_test, axis=-1)

# Normalizzazione dei dati
x_mnist_train = x_mnist_train / 255.0
x_mnist_test = x_mnist_test / 255.0

# Conversione etichette in one-hot encoding
y_mnist_train = to_categorical(y_mnist_train, num_classes=num_classes)
y_mnist_test = to_categorical(y_mnist_test, num_classes=num_classes)

# Definizione delle trasformazioni per la Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10, # Rotazione massima di 10 gradi
    width_shift_range=0.1, # Spostamento orizzontale massimo del 10% della larghezza dell'immagine
    height_shift_range=0.1, # Spostamento verticale massimo del 10% dell'altezza dell'immagine
    zoom_range=0.1, # Zoom massimo del 10%
    shear_range=0.1, # Angolo di taglio massimo di 10 gradi
    horizontal_flip=False, # Non utilizzare riflessione orizzontale
    vertical_flip=False # Non utilizzare riflessione verticale
)
input_shape = (28, 28, 1)
# Addestramento del modello sul 80% dei tuoi numeri e sul 20% di MNIST
model = Sequential()
model.add(Input(shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

# Compilazione del modello
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

train_generator = datagen.flow(
    np.concatenate((x_tuoi_train, x_mnist_train), axis=0),
    np.concatenate((y_tuoi_train, y_mnist_train), axis=0),
    batch_size=32,
    shuffle=True
)

# Calcola il numero di step per epoch
num_train_samples = len(np.concatenate((x_tuoi_train, x_mnist_train), axis=0))
steps_per_epoch = num_train_samples // 32

loss_object = tf.keras.losses.CategoricalCrossentropy()

# Definizione dell'ottimizzatore
optimizer = tf.keras.optimizers.Adam()


def train_step(model, batch_x, batch_y):
    with tf.GradientTape() as tape:
        predictions = model(batch_x, training=True)
        loss = loss_object(batch_y, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

# Addestramento del modello con il data generator
num_epochs = 20
for epoch in range(num_epochs):
    print("Epoch", epoch+1, "/", num_epochs)
    for step in range(steps_per_epoch):
        # Ottieni il batch successivo dal generatore di dati
        batch_x, batch_y = next(train_generator)
        # Chiamata alla funzione di addestramento
        train_step(model, batch_x, batch_y)
        
# Valutazione del modello
model.evaluate(np.concatenate((x_tuoi_test, x_mnist_test), axis=0), 
                np.concatenate((y_tuoi_test, y_mnist_test), axis=0))

# Valutazione del modello
model.evaluate(np.concatenate((x_tuoi_test, x_mnist_test), axis=0), 
                np.concatenate((y_tuoi_test, y_mnist_test), axis=0))

# Salvataggio del modello
model.save("C:\\Users\\rosse\\Documents\\GitHub\\capolavoro\\cnn_numeri.keras")
print("Modello salvato")
