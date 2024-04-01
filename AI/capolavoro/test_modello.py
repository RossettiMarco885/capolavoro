import os
import numpy as np
import keras.preprocessing.image as image_utils
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


def load_and_prepare_image(filename):

    # Stampa il nome del file
    print(filename)

    # Carica l'immagine come stringa
    img_string = tf.io.read_file(filename)

    # Converte la stringa in un tensore di tipo immagine
    img = tf.image.decode_image(img_string, channels=1)

    # Controlla il tipo di dati dell'immagine
    if img.dtype != tf.float32 and img.dtype != tf.uint8:
        raise TypeError("L'immagine deve essere in formato a virgola mobile o intero")

    # Se l'immagine è in uint8, convertila in float32
    if img.dtype == tf.uint8:
        img = tf.image.convert_image_dtype(img, tf.float32)

    # Ridimensiona l'immagine a 28x28
    img = tf.image.resize(img, [28, 28])

    # Converte l'immagine in un array NumPy
    img_np = img.numpy()

    # Denoising con filtro bilaterale
    img_np = cv2.bilateralFilter(img_np, 4, 75, 75)

    # Normalizza i dati dei pixel (supponendo un intervallo di 0-255)
    img_np = img_np / 255.0

    # Inverti i colori dell'immagine
    img_np = 1 - img_np

    # Reshape in un singolo campione con 1 canale
    img_np = np.reshape(img_np, [1, 28, 28, 1])

    return img_np


def carica_immagini(images):
        # Itera sui file nella directory
    for filename in os.listdir(path_images):
        # Carica e pre-processa l'immagine
        image = load_and_prepare_image(os.path.join(path_images, filename))

        # Se l'immagine è stata caricata correttamente
        if image is not None:
            images.append(image)
    return images

def mostra_immagini(images,path_images):
    for i in range(len(images)):
        image = load_and_prepare_image(
            os.path.join(path_images, os.listdir(path_images)[i])
        )

        # Visualizza le prime 5 immagini postprocessate
        if i < 5:
            plt.figure()
            plt.imshow(
                image.squeeze(), cmap="gray"
            )  # Squeeze per rimuovere le dimensioni 1
            plt.title(f"Immagine {i+1}")
            plt.axis("off")
            plt.show()


# Percorso della directory delle immagini
path_images = "D:\\AI\\capolavoro\\immagini\\"

# Carica il modello Keras pre-allenato
model = load_model("mnist_cnn.keras")

# Lista per memorizzare le immagini pre-processate
images = []

images = carica_immagini(images)

# ...
mostra_immagini(images,path_images)
# Dopo aver caricato e preparato l'immagine nel ciclo for


# Predice i numeri per tutte le immagini in un batch
predictions = model.predict(np.vstack(images))

predizioni_giuste = 0
predizioni_sbagliate = 0

# Itera sulle immagini e le loro predizioni
for i in range(len(images)):
    filename = os.listdir(path_images)[i]
    prediction = model.predict(images[i])
    numero_predetto = np.argmax(prediction)

    # Controllo se il numero predetto è uguale al numero nel nome del file
    numero_nel_nome = int(filename[0])
    if numero_nel_nome != int(numero_predetto):
        predizioni_sbagliate += 1
        print(
            f"Errore: {filename} --> {numero_nel_nome}, il numero predetto è {numero_predetto}"
        )
    else:
        predizioni_giuste += 1
        print(
            f"GIUSTO: {filename} --> {numero_nel_nome}, il numero predetto è {numero_predetto}"
        )
print(
    f"le predizioni giuste sono: {predizioni_giuste} \nle predizioni sbagliate sono: {predizioni_sbagliate}\nla percentuale di predizioni corrette è: {predizioni_giuste/predizioni_sbagliate*100}"
)
