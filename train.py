import csv
import cv2
import numpy as np

lines = []

with open ('data/driving_log.csv') as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    file_name = source_path.split('/')[-1]
    current_path = 'data/IMG/' + file_name
    image = cv2.imread(current_path)
    images.append(image) 
    measurement = float(line[4])
    measurements.append(measurement)

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D

model = Sequential()

# Normalizing lambda layer
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

# First convolutional layer
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())

# Second convolutional layer
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())

# Fully-connected layers
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle= True, nb_epoch=6)

model.save('model.h5')