# -*- coding: UTF-8 -*-
"""
# WANGZHE12
"""
import csv
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Activation, Dropout, Cropping2D


images = []
measurements = []
line_num = 0

dictory_list = ["./data", "./examples", "./examples2", "./examples3"]

for dictory in dictory_list:

    lines = []
    with open(dictory + '/driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            lines.append(line)

    for line in lines:
        if not line_num:
            line_num += 1
            continue
        measurement = float(line[3])
        if not measurement:
            continue
        if "\\" in line[0]:
            source_path0 = line[0].split("\\")[-1]
            source_path1 = line[0].split("\\")[-1]
            source_path2 = line[0].split("\\")[-1]
        else:
            source_path0 = line[0].split("/")[-1]
            source_path1 = line[0].split("/")[-1]
            source_path2 = line[0].split("/")[-1]
        
        filename = dictory + "/IMG/" + source_path0
        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # crop the image
        images.append(image)
        measurements.append(measurement)
        reverse_image = cv2.flip(image, 1)
        images.append(reverse_image)
        reverse_measurement = -1 * float(measurement)
        measurements.append(reverse_measurement)
        
#         filename = dictory + "/IMG/" + source_path1
#         image = cv2.imread(filename)
#         # crop the image
#         images.append(image)
#         measurements.append(measurement + 0.2)
#         reverse_image = cv2.flip(image, 1)
#         images.append(reverse_image)
#         reverse_measurement = -1 * float(measurement + 0.2)
#         measurements.append(reverse_measurement)
        
#         filename = dictory + "/IMG/" + source_path2
#         image = cv2.imread(filename)
#         # crop the image
#         images.append(image)
#         measurements.append(measurement - 0.2)
#         reverse_image = cv2.flip(image, 1)
#         images.append(reverse_image)
#         reverse_measurement = -1 * float(measurement - 0.2)
#         measurements.append(reverse_measurement)
    
print(len(images))
print(len(measurements))
    
X_train = np.array(images)
y_train = np.array(measurements)


model = Sequential()
model.add(Lambda(lambda x: x / 255.0, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Convolution2D(24, 5, 5, subsample=(2,2)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
# model.add(Dropout(0.5))
model.add(Convolution2D(36, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))

# model.add(Convolution2D(16, 5, 5, border_mode='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))

# model.add(Convolution2D(36, 5, 5, border_mode='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))

# model.add(Convolution2D(128, 5, 5, border_mode='same'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss="mse", optimizer='adam')

model.summary()

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

model.save("model2.h5")
