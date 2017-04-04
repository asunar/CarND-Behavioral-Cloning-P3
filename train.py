
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

# def get_images_measurements(lines):
#     images = []
#     measurements = []

#     for line in lines:
#         source_path = line[0]
#         tokens = source_path.split('/')
#         filename = tokens[-1]
#         local_path = "./data/IMG/" + filename
#         image = cv2.imread(local_path)
#         images.append(image)
#         measurement = line[3]
#         measurements.append(measurement)
    
#     print("Images count:" + str(len(images)))

    # images_array = np.array(images[1:])
    # measurements_array = np.array(measurements[1:])    
    # images_measurements = (images_array, measurements_array)   

    # return images_measurements 



def augment_images(images, measurements):
    augmented_images = []
    augmented_measurements = []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        flipped_image = cv2.flip(image, 1)
        flipped_measurement = float(measurement) * -1.0
        augmented_images.append(flipped_image)
        augmented_measurements.append(flipped_measurement)

    augmented_images_array = np.array(augmented_images)
    augmented_measurements_array = np.array(augmented_measurements)

    return (augmented_images_array, augmented_measurements_array) 

def get_left_right_camera_images_measurements_with_correction(lines):
    images = []
    measurements = []
    for line in lines:
        for i in range(3):
            # Load images from center, left and right cameras
            source_path = line[i]
            tokens = source_path.split('/')
            filename = tokens[-1]
            local_path = "./data/IMG/" + filename
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

    print("Lines count:" + str(len(lines)))
    print("Measurements count:" + str(len(measurements)))

    assert len(lines*3) == len(measurements), 'Actual number of *jpg images does not match with the csv log file'
    return (images, measurements)

def get_training_data(lines):
    images = []
    measurements = []
    for line in lines:
        for i in range(3):
            # Load images from center, left and right cameras
            source_path = line[i]
            tokens = source_path.split('/')
            filename = tokens[-1]
            local_path = "./data/IMG/" + filename
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

    # corrected_side_images = get_left_right_camera_images_measurements_with_correction(lines)
    # augmented_images_measurements = augment_images(corrected_side_images[0], corrected_side_images[1])

    # X_train = np.concatenate(augmented_images_measurements[0],
    #                 corrected_side_images[0])
    # y_train = np.concatenate(augmented_images_measurements[1],
    #                 corrected_side_images[1])

    return (X_train, y_train)

def build_model():
    model = Sequential()

    # Normalizing lambda layer
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))

    #crop
    model.add(Cropping2D(cropping=((70, 25), (1, 1))))

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

def train():
    logs = get_logs('data/driving_log.csv')
    training_data = get_training_data(logs)
    X_train, y_train = training_data[0], training_data[1]
    model = build_model()

    run_model(model, X_train, y_train)

train()
