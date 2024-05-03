import os
import numpy as np
from tensorflow.keras.utils import to_categorical

# Percorso alla tua cartella con le immagini dei numeri scritti a mano
handwritten_numbers_dir = "C:\\Users\\rosse\\Documents\\GitHub\\capolavoro\\IMG_ADDESTRAMENTO\\"

# Lista per memorizzare le etichette e i nomi dei file
labels = []
filenames = []

# Scansione dei file nella cartella
for filename in os.listdir(handwritten_numbers_dir):
    if filename.endswith(".png"):
        # Estrai l'etichetta dal nome del file
        label = int(filename[0])
        labels.append(label)
        filenames.append(os.path.join(handwritten_numbers_dir, filename))

# Numero totale di classi
num_classes = 10

# Converti le etichette in one-hot encoding
y_labels = to_categorical(labels, num_classes=num_classes)

# Stampa le prime 5 etichette e i relativi nomi dei file per verificare
print("Primi 5 esempi:")
for i in range(100):
    print(f"Nome file: {filenames[i]}, Etichetta: {labels[i]}, One-hot encoding: {y_labels[i]}")
