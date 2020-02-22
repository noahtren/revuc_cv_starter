"""Simple video stream from webcam using OpenCV
"""

import cv2
import code
import argparse
import matplotlib.pyplot as plt

import tensorflow as tf
from neural_network import resnet

parser = argparse.ArgumentParser()
parser.add_argument('--int', action='store_true')
opt = parser.parse_args()

cap = cv2.VideoCapture(0)

def show(img):
    plt.imshow(img); plt.show()

while True:
    # get frame
    ret, frame = cap.read()
    # display frame
    cv2.imshow('frame', frame)

    frame = tf.cast(frame, tf.float32) / 255.
    frame = tf.stack((frame[:, :, 2], frame[:, :, 1], frame[:, :, 0]), axis=2)
    if opt.int: code.interact(local={**locals(), **globals()})

    frame = tf.expand_dims(frame, 0)
    frame = tf.image.resize(frame, (224, 224))
    if opt.int: code.interact(local={**locals(), **globals()})

    raw_prediction = resnet.predict(frame)
    if opt.int: code.interact(local={**locals(), **globals()})

    prediction = tf.keras.applications.imagenet_utils.decode_predictions(raw_prediction)
    if opt.int: code.interact(local={**locals(), **globals()})

    print(prediction)

    # exit if user enters 'q' key on keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# close webcam
cap.release()
cv2.destroyAllWindows()
