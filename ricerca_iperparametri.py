from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np

# Function to create the model
def create_model(optimizer='adam'):
  model = Sequential()
  model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 3), kernel_regularizer=l2(0.01)))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))
  model.add(Conv2D(64, kernel_size=(3, 3), activation="relu", kernel_regularizer=l2(0.01)))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(256, activation="relu", kernel_regularizer=l2(0.01)))
  model.add(Dropout(0.5))
  model.add(Dense(num_classes, activation="softmax"))
  model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])
  return model

# Define the model for GridSearchCV
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define hyperparameter grid
param_grid = {'batch_size': [32, 64],
              'epochs': [10, 20],
              'optimizer': ['adam', 'rmsprop']}

# Perform GridSearchCV
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_result = grid.fit(np.concatenate((x_tuoi_train, x_mnist_train), axis=0), 
                      np.concatenate((y_tuoi_train, y_mnist_train), axis=0))

# Print results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
