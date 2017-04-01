import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D


def get_logs(path):
    lines = []
    with open (path) as csvfile:
        next(csvfile)
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    return lines

def get_training_data(lines):
    images = []
    measurements = []

    for line in lines:
        source_path = line[0]
        tokens = source_path.split('/')
        filename = tokens[-1]
        local_path = "./data/IMG/" + filename
        image = cv2.imread(local_path)
        images.append(image)
        measurement = line[3]
        measurements.append(measurement)
    
    print("Images count:" + str(len(images)))

    X_train = np.array(images[1:])
    y_train = np.array(measurements[1:])    
    training_data = (X_train, y_train)
    return training_data

# augmented_images = []
# augmented_measurements = []
# for image, measurement in zip(images, measurements):
# 	augmented_images.append(image)
# 	augmented_measurements.append(measurement)
	
# 	# Flip images to reduce bias from anti-clockwise driving
# 	flipped_image = cv2.flip(image, 1)
# 	flipped_measurement = float(measurement) * -1.0
# 	augmented_images.append(flipped_image)
# 	augmented_measurements.append(flipped_measurement)

# X_train = np.array(augmented_images)
# y_train = np.array(augmented_measurements)

# print("Augmented Images count:" + str(len(augmented_images)))

def build_model():
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

    print("Built the model")
    return model

def run_model(model, X_train,y_train):
    model.fit(X_train, y_train, validation_split=0.2, shuffle= True, nb_epoch=6)
    print("Model fit complete, saving model")
    model.save('model.h5')

logs = get_logs('data/driving_log.csv')
training_data = get_training_data(logs)
X_train, y_train = training_data[0], training_data[1]
model = build_model()

run_model(model, X_train, y_train)