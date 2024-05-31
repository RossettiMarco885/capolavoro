import tkinter as tk
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from PIL import Image, ImageGrab, ImageTk
import tkinter.messagebox as mb

# Carica il modello addestrato
model = load_model("cnn_numeri.keras")
g_confidenza = 26
g_predizione = 30

# Variabili globali per tracciare la posizione precedente del mouse
last_x, last_y = None, None

# Funzione per iniziare il disegno
def paint_start(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y

# Funzione per gestire il disegno sul canvas
def paint(event):
    global last_x, last_y
    if last_x and last_y:
        canvas.create_line(last_x, last_y, event.x, event.y, fill="black", width=40, capstyle=tk.ROUND, smooth=tk.TRUE)
    last_x, last_y = event.x, event.y

# Funzione per fermare il disegno
def paint_stop(event):
    global last_x, last_y
    last_x, last_y = None, None

# Funzione per pulire il canvas
def clear_canvas():
    canvas.delete("all")
    global last_x, last_y
    last_x, last_y = None, None

# Funzione per prevedere il numero dall'immagine disegnata
def predict_number():
    try:
        x0 = canvas.winfo_rootx() + canvas.winfo_x()
        y0 = canvas.winfo_rooty() + canvas.winfo_y()
        x1 = x0 + canvas.winfo_width()
        y1 = y0 + canvas.winfo_height()
        
        # Acquisizione dell'immagine dalla canvas
        img = ImageGrab.grab((x0, y0, x1, y1))
        
        # Converti l'immagine in grayscale e ridimensionala a 28x28
        img = img.convert('L')
        img = img.resize((28, 28), Image.LANCZOS)
        
        # Converti l'immagine in un array NumPy
        img = np.array(img)
        
        # Inverti i colori (MNIST è bianco su nero)
        img = 255 - img
        
        # Reshape dell'immagine in un singolo campione con 1 canale
        img = img.reshape(1, 28, 28, 1)
        
        # Normalizzazione dei dati dei pixel
        img = img.astype("float32") / 255.0
        
        # Previsione del numero con il modello
        prediction = model.predict(img)
        #predicted_number = np.argmax(prediction)

        # Calcolo delle percentuali di confidenza
        confidence_scores = prediction[0]
        top_3_indices = confidence_scores.argsort()[-3:][::-1]
        top_3_confidences = confidence_scores[top_3_indices] * 100

        # Visualizzazione del numero predetto in rosso e più grande
        predicted_label.config(text=f"Numero predetto: {top_3_indices[0]}", fg="red", font=("Helvetica", g_predizione, "bold"))

        # Visualizzazione delle percentuali di confidenza
        confidence_text = (f"Confidenza: {top_3_confidences[0]:.2f}% (Numero: {top_3_indices[0]})\n"
                           f"Seconda confidenza: {top_3_confidences[1]:.2f}% (Numero: {top_3_indices[1]})\n"
                           f"Terza confidenza: {top_3_confidences[2]:.2f}% (Numero: {top_3_indices[2]})")
        precision_label.config(text=confidence_text, fg="red", font=("Helvetica", g_confidenza, "bold"))
        
    except Exception as e:
        mb.showerror("Errore", f"Si è verificato un errore: {e}")

# Crea la finestra principale della GUI
root = tk.Tk()
root.title("Riconoscimento Numeri")

# Massimizza la finestra
root.state('zoomed')

# Carica l'immagine di sfondo
background_image = Image.open("sfondo/esagonale.jpg")
background_photo = ImageTk.PhotoImage(background_image)

# Crea una label per l'immagine di sfondo
background_label = tk.Label(root, image=background_photo)
background_label.place(relwidth=1, relheight=1)

# Definisci il canvas per il disegno
canvas = tk.Canvas(root, width=700, height=700, bg="white", borderwidth=0)
canvas.grid(row=0, column=0, rowspan=5, padx=10, pady=10)

# Collega gli eventi del mouse per disegnare sul canvas
canvas.bind("<Button-1>", paint_start)
canvas.bind("<B1-Motion>", paint)
canvas.bind("<ButtonRelease-1>", paint_stop)

# Button per predire il numero
predict_button = tk.Button(root, text="Predici Numero", command=predict_number, bg="#4CAF50", fg="white", font=("Helvetica", 14, "bold"))
predict_button.grid(row=2, column=1, padx=10, pady=10)

# Button per pulire il canvas
clear_button = tk.Button(root, text="Pulisci", command=clear_canvas, bg="#9E9E9E", fg="white", font=("Helvetica", 14, "bold"))
clear_button.grid(row=1, column=1, padx=10, pady=10)

# Button per chiudere la finestra
close_button = tk.Button(root, text="Chiudi", command=root.destroy, bg="#f44336", fg="white", font=("Helvetica", 14, "bold"))
close_button.grid(row=0, column=1, padx=10, pady=10)

# Label per visualizzare il numero predetto
predicted_label = tk.Label(root, text="Disegna un Numero", fg="red", font=("Helvetica", g_predizione, "bold"))
predicted_label.grid(row=3, column=1, padx=10, pady=10)

# Label per visualizzare la confidenza predetta
precision_label = tk.Label(root, text="Il programma cercherà di capire che numero è", fg="red", font=("Helvetica", g_confidenza, "bold"))
precision_label.grid(row=4, column=1, padx=10, pady=10)

root.mainloop()
