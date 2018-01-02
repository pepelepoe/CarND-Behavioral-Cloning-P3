import csv
import cv2
import numpy as np

lines = []
with open('/home/pepelepoe/Downloads/linux_sim/USelfTrainingData/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

featureset = [] #image inputs
labelset = []   #measurement output
for line in lines[1:]:
    # img_path = line[0]
    # filename = img_path.split('/')[-1]
    image = cv2.imread(line[0])
    # image = np.reshape((image, 160, 320, 3))
    featureset.append(image)
    measurement = float(line[3])
    labelset.append(measurement)

X_train = np.array(featureset)
y_train = np.array(labelset)

# print(X_train)
print('X_train shape:', X_train.shape)

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(160,320,3)))
model.add(Dense(1))
#regression model
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True)

model.save('model.h5')
