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

print("Images count:" + str(len(images)))

augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
	augmented_images.append(image)
	augmented_measurements.append(measurement)
	
	# Flip images to reduce bias from anti-clockwise driving
	flipped_image = cv2.flip(image, 1)
	flipped_measurement = float(measurement) * -1.0
	augmented_images.append(flipped_image)
	augmented_measurements.append(flipped_measurement)

X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

print("Augmented Images count:" + str(len(augmented_images)))

def build_model():
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

    print("Built the model")
    return model

def run_model(model, X_train,y_train):
    model.fit(X_train, y_train, validation_split=0.2, shuffle= True, nb_epoch=6)
    print("Model fit complete, saving model")
    model.save('model.h5')

# run_model(build_model(), X_train, y_train)