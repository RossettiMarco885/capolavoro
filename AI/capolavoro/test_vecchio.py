import os
import numpy as np
import keras.preprocessing.image as image_utils
from keras.models import load_model
import matplotlib.pyplot as plt

def load_and_prepare_image(filename):
    img = image_utils.load_img(filename, color_mode="grayscale", target_size=(28, 28))
    img = image_utils.img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype("float32") / 255.0
    return img

def carica_immagini(images):
    for filename in os.listdir(path_images):
        image = load_and_prepare_image(os.path.join(path_images, filename))
        if image is not None:
            images.append(image)
    return images

def mostra_immagini(images):
    #for i in range(len(images)):
     for i in range(0):
        plt.figure()
        plt.imshow(images[i].squeeze(), cmap="gray")
        plt.title(f"Immagine {i+1}")
        plt.axis("off")
        plt.show()

def predict_from_image(image):
    prediction = model.predict(image)
    numero_predetto = np.argmax(prediction)
    return numero_predetto

# Percorso della directory delle immagini di Paint
path_images = "C:\\Users\\rosse\\Documents\\GitHub\\capolavoro\\AI\\capolavoro\\immagini_grandi\\"

# Carica il modello pre-allenato
model = load_model("mnist_cnn.keras")

# Lista per memorizzare le immagini pre-processate
images = []

# Carica le immagini e le pre-processa
images = carica_immagini(images)

# Visualizza le immagini pre-processate
mostra_immagini(images)

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

# Imposta il valore di predizioni sbagliate a 1 se è zero per evitare divisione per zero
if predizioni_sbagliate == 0:
    predizioni_sbagliate = 1
print(f"La percentuale di predizioni corrette è: {(predizioni_giuste / predizioni_sbagliate) * 100}")
