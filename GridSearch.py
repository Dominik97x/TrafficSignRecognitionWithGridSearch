from __future__ import print_function
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from PIL import Image
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data = []
labels = []
classes = 43
cur_path = os.getcwd()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# Retrieving the images and their labels
for i in range(classes):
    path = os.path.join(cur_path, 'train', str(i))
    images = os.listdir(path)

    for a in images:
        try:
            image = Image.open(path + '\\' + a)
            image = image.resize((30, 30))
            image = np.array(image)
            data.append(image)
            labels.append(i)
        except:
            print("Error loading image")

# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)
print(data.shape, labels.shape)
# Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# Converting the labels into one hot encoding
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)


def create_model(filter_1,filter_2,kernel_1,kernel_2,dense_1):
    model = Sequential()
    model.add(Conv2D(filter_1, kernel_1, activation='relu', input_shape=X_train.shape[1:]))
    model.add(Conv2D(filter_1, kernel_1, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filter_2, kernel_2, activation='relu'))
    model.add(Conv2D(filter_2, kernel_2, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(dense_1, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))
    # Kompilacja modelu
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


param_grid = {
    'filter_1': [16, 32, 64],
    'filter_2': [32, 48, 64],
    'kernel_1': [4, 5, 6],
    'kernel_2': [2, 3, 4],
    'dense_1': [196, 256, 320]
}


my_classifier = KerasClassifier(build_fn=create_model, epochs=10, batch_size=32)
grid = GridSearchCV(my_classifier, param_grid, cv=3, n_jobs=1, verbose=1)
grid_result = grid.fit(X_train, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))




# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# print("Test Accuracy", grid_result.score(X_test, y_test))
