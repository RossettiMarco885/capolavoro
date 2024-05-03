import os
import numpy as np
import keras.preprocessing.image as image_utils
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


def load_and_prepare_image(filename):
    """
    Carica e pre-processa un'immagine di un numero scritto a mano.

    Args:
        filename: Il nome del file immagine.

    Returns:
        L'immagine pre-processata in formato NumPy.
    """

    # Carica l'immagine in modalità grayscale con dimensione target di 28x28
    img = image_utils.load_img(filename, color_mode="grayscale", target_size=(28, 28))

    # Converte l'immagine in un array NumPy
    img = image_utils.img_to_array(img)

    # Denoising con filtro bilaterale
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # Reshape in un singolo campione con 1 canale
    img = img.reshape(1, 28, 28, 1)

    # Normalizza i dati dei pixel (supponendo un intervallo di 0-255)
    # img = 1 - img
    img = img.astype("float32") / 255.0

    return img


def carica_immagini(images):
    """
    Carica le immagini dalla directory specificata e le pre-processa.

    Args:
        images: Lista per memorizzare le immagini pre-processate.

    Returns:
        La lista di immagini pre-processate.
    """

    # Itera sui file nella directory
    for filename in os.listdir(path_images):
        # Carica e pre-processa l'immagine
        image = load_and_prepare_image(os.path.join(path_images, filename))

        # Se l'immagine è stata caricata correttamente
        if image is not None:
            images.append(image)
    return images


def mostra_immagini(images, path_images):
    """
    Visualizza le prime n immagini pre-processate.

    Args:
        images: La lista di immagini pre-processate.
        path_images: Il percorso della directory delle immagini.
    """

    #modificare n con il numero di cifre massimo in un numero per vederne una per numero
    #es. 77777 --> 5 cifre e quindi n=5
    #n = 5
    for i in range(len(images)):

        # Visualizza le prime n immagini postprocessate
        #if i < 1000 and i % n == 0:
         if i<0:
            plt.figure()
            plt.imshow(
                images[i].squeeze(), cmap="gray"
            )  # Squeeze per rimuovere le dimensioni 1
            plt.title(f"Immagine {i+1}")
            plt.axis("off")
            plt.show()


def predict_from_image(image):
    """
    Funzione per predire il numero dall'immagine.

    Args:
        image: L'immagine pre-processata in formato NumPy.

    Returns:
        Il numero predetto dall'immagine.
    """

    # Predizione del numero
    prediction = model.predict(image)
    numero_predetto = np.argmax(prediction)

    return numero_predetto


# Percorso della directory delle immagini
path_images = "C:\\Users\\rosse\\Documents\\GitHub\\capolavoro\\IMG_TEST\\"

# Carica il modello Keras pre-allenato
model = load_model("mnist_numeri_miei.keras")

# Lista per memorizzare le immagini pre-processate
images = []

# Carica le immagini e le pre-processa
images = carica_immagini(images)

# Visualizza le prime 5 immagini pre-processate
mostra_immagini(images, path_images)

# Analisi delle predizioni

predizioni_giuste = 0
predizioni_sbagliate = 0
errori_per_numero = {numero: 0 for numero in range(10)}
total_per_numero = {numero: 0 for numero in range(10)}

for i in range(len(images)):
    filename = os.listdir(path_images)[i]
    numero_nel_nome = int(filename[0])
    numero_predetto = predict_from_image(images[i])

    if numero_nel_nome != numero_predetto:
        predizioni_sbagliate += 1
        errori_per_numero[numero_nel_nome] += 1

        print(f"Errore: {filename} --> {numero_nel_nome}, il numero predetto è {numero_predetto}")
    else:
        predizioni_giuste += 1
        print(f"GIUSTO: {filename} --> {numero_nel_nome}, il numero predetto è {numero_predetto}")

    total_per_numero[numero_nel_nome] += 1

# Stampa il numero totale di predizioni corrette e sbagliate per ciascun numero
for numero in range(10):
    total_predizioni = total_per_numero[numero]
    total_corrette = total_predizioni - errori_per_numero[numero]
    total_sbagliate = errori_per_numero[numero]

    print(f"Numero {numero}: Totali={total_predizioni}, Corrette={total_corrette}, Sbagliate={total_sbagliate}")

print(f"Le predizioni giuste sono: {predizioni_giuste}")
print(f"Le predizioni sbagliate sono: {predizioni_sbagliate}")

print(f"La percentuale di predizioni corrette è: {(predizioni_giuste / (predizioni_giuste+predizioni_sbagliate)) * 100}")


