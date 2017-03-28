import csv
import cv2
import numpy as np

lines = []

with open ('data/data/driving_log.csv') as csvfile:
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


print(y_train[100])