import csv
import cv2
import numpy as np

lines = []
with open('/home/pepelepoe/Code/USelfDriving/USelfTrainingData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

featureset = [] #image inputs
labelset = []   #measurement output
for line in lines:
    for i in range(3):
        image = cv2.imread(line[0])
        # if image == None:
        #     print("Incorrect path", line[0])
        featureset.append(image)
        measurement = float(line[3])
        labelset.append(measurement)

augImg, augLbl = [], []
for image, label in zip(featureset, labelset):
    augImg.append(image)
    augLbl.append(label)
    augImg.append(cv2.flip(image, 1))
    augLbl.append(label*-1.0)

Xtrain = np.array(augImg)
ytrain = np.array(augLbl)

# print(X_train)
print('X_train shape:', Xtrain.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
#normalize (divide each element by 255) to a range of 0 and 1 and mean center image (substract 0.5)
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70, 25),(0, 0))))

# LENET
# model.add(Convolution2D(6,5,5,activation="relu"))
# model.add(MaxPooling2D())
# # model.add(Dropout(0.5))
# model.add(Convolution2D(6,5,5,activation="relu"))
# model.add(MaxPooling2D())
# # model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(1))

#NVIDIA
model.add(Convolution2D(24,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2), activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#regression model
model.compile(loss='mse', optimizer='adam')
model.fit(Xtrain, ytrain, validation_split=0.2, shuffle=True, nb_epoch=2)

model.save('model.h5')
