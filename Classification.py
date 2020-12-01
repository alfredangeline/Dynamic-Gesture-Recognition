import cv2
from keras.models import load_model
import keras
import numpy as np
import pandas as pd
import os

data_root = os.getcwd()

path_model = data_root + '/Model/resnetmodel.hdf5'
model = load_model(path_model)

path_labels = data_root + '/jester-v1-labels.csv'
labels = pd.read_csv(path_labels, names=['label'], header=None)

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)

buffer = []
predicted_value = 0
final_label = ''
i = 1
while video.isOpened():
    ret, frame = video.read()
    if ret:
        image = cv2.resize(frame, (96,64))
        image = image/255
        buffer.append(image)
        if(i%16 == 0):
            buffer = np.expand_dims(buffer, 0)
            predicted_value = np.argmax(model.predict(buffer))
            cls = labels.iloc[predicted_value]
            print(cls)
            print(cls.iloc[0])

            final_label = labels.label[predicted_value]

            cv2.imshow('frame', frame)
            buffer = []
        i = i + 1
        text = 'activity: {}'.format(final_label)
        cv2.putText(frame, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.15, (0, 255, 0), 5)
        cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
