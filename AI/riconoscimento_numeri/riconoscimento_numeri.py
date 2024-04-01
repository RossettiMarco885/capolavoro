import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from extra_keras_datasets.emnist import load_data
import os

# Caricamento dei dataset
# (x_train_emnist, y_train_emnist), (x_test_emnist, y_test_emnist) = load_data()
# (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

# Preprocessing dei dati
##x_train = tf.concat([x_train_mnist, x_train_emnist], axis=0)
# y_train = tf.concat([y_train_mnist, y_train_emnist], axis=0)
# x_test = tf.concat([x_test_mnist, x_test_emnist], axis=0)
# y_test = tf.concat([y_test_mnist, y_test_emnist], axis=0)

(x_train, y_train), (x_test, y_test) = load_data()

x_train = tf.cast(x_train, tf.float32) / 255.0
x_test = tf.cast(x_test, tf.float32) / 255.0


y_train = to_categorical(y_train, 62)
y_test = to_categorical(y_test, 62)

# Definizione del modello
cnn = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(62, activation="softmax"),
    ]
)

# Compilazione del modello
cnn.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Addestramento del modello
callbacks = [tf.keras.callbacks.EarlyStopping(patience=5)]
history = cnn.fit(
    x_train, y_train, epochs=100, validation_split=0.2, callbacks=callbacks
)

# Valutazione del modello
test_loss, test_acc = cnn.evaluate(x_test, y_test, verbose=2)
print("\nTest accuracy:", test_acc)

# Visualizzazione delle curve di apprendimento
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.show()

# Matrice di confusione
y_pred = cnn.predict(x_test)
y_pred_classes = tf.argmax(y_pred, axis=1)
cm = tf.math.confusion_matrix(tf.argmax(y_test, axis=1), y_pred_classes)

plt.figure(figsize=(10, 10))
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.show()

# Predizione su immagini
image_dir = "D:\\AI\\riconoscimento_numeri\\immagini"
filenames = os.listdir(image_dir)

for filename in filenames:
    image = tf.keras.preprocessing.image.load_img(
        os.path.join(image_dir, filename), color_mode="grayscale", target_size=(28, 28)
    )
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = input_arr / 255.0
    input_arr = tf.expand_dims(input_arr, axis=0)

    predictions = cnn.predict(input_arr)
    predicted_class = tf.argmax(predictions, axis=1)

    plt.imshow(np.squeeze(input_arr), cmap="gray")
    plt.title(f"Predicted class: {predicted_class[0]}")
    plt.show()


cnn.save("D:\\AI\\riconoscimento_numeri\\modello_riconoscimento_numeri_emnist.keras")
print("salvato")
