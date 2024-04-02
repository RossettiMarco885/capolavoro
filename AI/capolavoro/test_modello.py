import os
import numpy as np
import keras.preprocessing.image as image_utils
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2


def load_and_prepare_image(filename):

     # Carica l'immagine in modalità grayscale con dimensione target di 28x28
  img = image_utils.load_img(filename, color_mode="grayscale", target_size=(28, 28))

  # Converte l'immagine in un array NumPy
  img = image_utils.img_to_array(img)

  # Reshape in un singolo campione con 1 canale
  img = img.reshape(1, 28, 28, 1)

  # Normalizza i dati dei pixel (supponendo un intervallo di 0-255)
  #img = 1 - img
  img = img.astype("float32") / 255.0
  

  return img


def carica_immagini(images):
    # Itera sui file nella directory
    for filename in os.listdir(path_images):
        # Carica e pre-processa l'immagine
        image = load_and_prepare_image(os.path.join(path_images, filename))

        # Se l'immagine è stata caricata correttamente
        if image is not None:
            images.append(image)
    return images


def mostra_immagini(images, path_images):
    for i in range(len(images)):

        # Visualizza le prime 5 immagini postprocessate
        if i < 1:
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
path_images = "D:\\AI\\capolavoro\\immagini\\"

# Carica il modello Keras pre-allenato
model = load_model("mnist_cnn.keras")

# Lista per memorizzare le immagini pre-processate
images = []

# Carica le immagini e le pre-processa
images = carica_immagini(images)

# Visualizza le prime 5 immagini pre-processate
mostra_immagini(images, path_images)

# Predice i numeri per tutte le immagini in un batch
predictions = model.predict(np.vstack(images))

predizioni_giuste = 0
predizioni_sbagliate = 0

"""
# Itera sulle immagini e le loro predizioni
for i in range(len(images)):
    filename = os.listdir(path_images)[i]
    prediction = model.predict(images[i])
    numero_predetto = np.argmax(prediction)

    # Controllo se il numero predetto è uguale al numero nel nome del file
    numero_nel_nome = int(filename[0])
    if numero_nel_nome != int(numero_predetto):
"""


for i in range(len(images)):
        filename = os.listdir(path_images)[i]
        numero_predetto=predict_from_image(images[i])

        # Controllo se il numero predetto è uguale al numero nel nome del file
        numero_nel_nome = int(filename[0])
        if numero_nel_nome != numero_predetto:
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
        f"le predizioni giuste sono: {predizioni_giuste} \nle predizioni sbagliate sono: {predizioni_sbagliate}\nla percentuale di predizioni corrette è: {(predizioni_giuste/predizioni_sbagliate)*100}"
    )   
