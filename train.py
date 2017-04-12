
import cv2

import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D,Cropping2D


def get_logs(path):
    import csv
    lines = []
    with open (path) as csvfile:
        next(csvfile)
        reader = csv.reader(csvfile)
        return [line for line in reader]

def copy_my_images_to_data():
    import os
    import shutil
    destination = 'data/IMG'
    # files = os.listdir('/etc/BNALP')
    my_training_data_path = 'my_training_data/IMG'
    files = os.listdir(my_training_data_path)
    for file in files:
        shutil.move(my_training_data_path + "/" + file, destination)

def get_training_data(lines, local_image_path):
    images = []
    measurements = []
    counter = 0
    for line in lines:
        if counter == 10:
            break
        for i in range(3):
            # Load images from center, left and right cameras
            source_path = line[i]
            tokens = source_path.split('/')
            filename = tokens[-1]
            local_path = local_image_path + filename
            image = cv2.imread(local_path)
            images.append(image)
        correction = 0.2
        measurement = float(line[3])
        
        # Steering adjustment for center images
        measurements.append(measurement)
        
        # Add correction for steering for left images
        measurements.append(measurement+correction)
        
        # Minus correction for steering for right images
        measurements.append(measurement-correction)
        counter = counter + 1

    augmented_images = []
    augmented_measurements = []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        flipped_image = cv2.flip(image, 1)
        flipped_measurement = float(measurement) * -1.0
        augmented_images.append(flipped_image)
        augmented_measurements.append(flipped_measurement)

    X_train = np.array(augmented_images)
    y_train = np.array(augmented_measurements)    

    print("X_train:" + str(len(X_train)))
    print("y_train:" + str(len(y_train)))

    return (X_train, y_train)

def build_model():
    model = Sequential()

    #Nvidia model
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(1,1))))
    model.add(Convolution2D(24,5,5, subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')

    print("Built the model")
    return model

def run_model(model, X_train,y_train):
    model.fit(X_train, y_train, validation_split=0.2, shuffle= True, nb_epoch=3)
    print("Model fit complete, saving model")
    model.save('model.h5')

def train():
    sample_log = get_logs('data/driving_log.csv')
#    my_log = get_logs('my_training_data/driving_log.csv')

#    import itertools
#    all_logs = itertools.chain(sample_log, my_log)
    sample_training_data = get_training_data(sample_log, "./data/IMG/")
    # my_training_data = get_training_data(sample_log, "./my_training_data/IMG/")
    X_train, y_train = (sample_training_data[0], sample_training_data[1])
    # X_train, y_train = (training_data[0], training_data[1])
    # model = build_model()

    # run_model(model, X_train, y_train)


# copy_my_images_to_data()
train()
