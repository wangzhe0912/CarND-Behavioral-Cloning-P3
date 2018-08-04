# -*- coding: UTF-8 -*-
"""
# WANGZHE12
"""
import csv
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Activation, Dropout


lines = []
with open('./data/driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

images = []
measurements = []
line_num = 0

for line in lines:
    if not line_num:
        line_num += 1
        continue
    source_path = line[0]
    filename = './data/' + source_path
    image = cv2.imread(filename)
    # crop the image
    image = image[55:135, :, :]
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    reverse_image = cv2.flip(image, 1)
    # plt.imshow(reverse_image)
    # plt.show()
    images.append(reverse_image)
    reverse_measurement = -1 * float(line[3])
    measurements.append(reverse_measurement)

X_train = np.array(images)
y_train = np.array(measurements)


model = Sequential()
model.add(Lambda(lambda x: x / 255.0, input_shape=(80, 320, 3)))


model.add(Convolution2D(6, 5, 5, border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(16, 5, 5, border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(36, 5, 5, border_mode='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(1))
model.compile(loss="mse", optimizer='adam')

model.summary()

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=20)

model.save("model.h5")
