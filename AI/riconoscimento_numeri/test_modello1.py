import keras.models as models
from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt


# Funzione per stampare il numero predetto e il secondo numero con probabilità maggiore
def stampa_max_secondo_max(predizione):
    # Mappatura delle classi alle etichette (sia numeri che lettere)
    classi = [str(i) for i in range(10)]  # Numeri da 0 a 9
    classi.extend([chr(i) for i in range(65, 91)])  # Lettere maiuscole da A a Z
    classi.extend([chr(i) for i in range(97, 123)])  # Lettere minuscole da a a z

    # Ottieni gli indici che ordinerebbero l'array in ordine crescente
    indici_ordinati = np.argsort(predizione[0])

    # Se ci sono meno di due classi predette, restituisci un messaggio di errore
    if len(indici_ordinati) < 2:
        return "Errore: meno di due classi predette"

    # Gli indici dell'array ordinato in ordine decrescente sono gli ultimi due elementi di 'indici_ordinati'
    max_indice = indici_ordinati[-1]
    secondo_max_indice = indici_ordinati[-2]
    arr_ordinato = sorted(predizione[0], reverse=True)  # nota l'aggiunta di [0] qui

    # Verifica se gli indici massimi sono all'interno dell'intervallo delle classi
    if 0 <= max_indice < len(classi) and 0 <= secondo_max_indice < len(classi):
        # Ottieni le etichette corrispondenti ai massimi indici di predizione
        max_etichetta = classi[max_indice]
        secondo_max_etichetta = classi[secondo_max_indice]
        return f"Classe Predetta: {max_etichetta} con probabilità: {arr_ordinato[0]},\nclasse con seconda probabilità maggiore ({secondo_max_etichetta}): {arr_ordinato[1]}"
    else:
        return "Errore: indice di classe predetta non valido"


# Carica il modello salvato
try:
    modello_caricato = models.load_model(
        "D:\\AI\\riconoscimento_numeri\\modello_riconoscimento_numeri.keras"
    )
except OSError as e:
    print(f"Errore durante il caricamento del modello: {e}")
    exit()

# Definisci la cartella contenente le immagini
cartella_immagini = "D:\\AI\\riconoscimento_numeri\\immagini"

# Definisci la cartella di output per le immagini post-processate
cartella_output = "D:\\AI\\riconoscimento_numeri\\immagini_postprocessate"

# Crea la cartella di output se non presente
if not os.path.exists(cartella_output):
    os.makedirs(cartella_output)

# Ottieni la lista di file nella cartella
files = os.listdir(cartella_immagini)

# Carica e processa tutte le immagini nella cartella
for file in files:
    # Costruisci il percorso completo del file
    percorso_file = os.path.join(cartella_immagini, file)

    # Carica l'immagine
    img = Image.open(percorso_file).convert("L")

    # Ridimensiona l'immagine
    img = img.resize((28, 28))

    # Inverti i colori dell'immagine (sfondo bianco e numeri neri)
    img = np.array(img)
    img = cv2.bitwise_not(img)

    # Centra l'immagine
    img_centrata = cv2.GaussianBlur(img, (3, 3), 0)
    # Controllo se la maschera è vuota
    if not img_centrata.any():
        # La maschera è vuota,
        # La maschera è vuota, calcola la media dell'intera immagine
        media = cv2.mean(img)[0]
    else:
        # La maschera non è vuota, converti in CV_8U e calcola la media
        img_centrata = img_centrata.astype(np.uint8)
        media = cv2.mean(img, mask=img_centrata)[0]
        # Binarizzazione adattiva con Otsu
        img_binaria = cv2.threshold(
            img_centrata, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

    # Converti l'array NumPy in un'immagine
    img_pil = Image.fromarray(img_binaria)

    # Applica il filtro antialiasing
    #img_pil = img_pil.filter(ImageFilter.SMOOTH)

    # Converti l'immagine di nuovo in un array NumPy
    img_binaria = np.array(img_pil)

    # Normalizza l'immagine
    img_array = img_binaria / 255.0

    # Reshape per la predizione
    img_array = img_array.reshape(1, 28, 28, 1)

    # Effettua la predizione
    predizione = modello_caricato.predict(img_array)

    # Stampa il titolo dell'immagine con la classe predetta e la probabilità
    plt.title(f"{stampa_max_secondo_max(predizione)}")

    # Mostra l'immagine a schermo
    plt.imshow(img, cmap="gray")
    plt.show()

    # Salva l'immagine post-processata
    nome_file = f"{file.split('.')[0]}_postprocessato.png"
    percorso_file_output = os.path.join(cartella_output, nome_file)
    img_pil.save(percorso_file_output)
