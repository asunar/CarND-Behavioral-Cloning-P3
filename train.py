
import cv2
import sklearn
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D
import sklearn.utils
from sklearn.model_selection import train_test_split

def get_logs(path):
    import csv

    samples = []
    with open(path) as csvfile:
        next(csvfile)
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    return (train_samples, validation_samples)
        
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        # shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

def get_training_data(lines, local_image_path):
    images = []
    measurements = []

    for line in lines:
        for i in range(3):
            # Load images from center, left and right cameras
            source_path = line[i]
            tokens = source_path.split('/')
            filename = tokens[-1]
            local_path = local_image_path + filename
            # print(local_path)
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

def run_model(model, train_samples, validation_samples):
    # model.fit(X_train, y_train, validation_split=0.2, shuffle= True, nb_epoch=6)
    # print("Model fit complete, saving model")
    # model.save('model.h5')

    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)

    model.fit_generator(train_generator, samples_per_epoch= 
                len(train_samples), validation_data=validation_generator, 
                nb_val_samples=len(validation_samples), nb_epoch=3)    
    model.save('model.h5')

def print_sample(list):
    result = [x[0] + " - " + x[3] for x in list]
    print(result) 

def train():
    train_samples, validation_samples = get_logs('data/driving_log.csv')

    model = build_model()
    run_model(model, train_samples, validation_samples)

train()

