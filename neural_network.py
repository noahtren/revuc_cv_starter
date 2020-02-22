"""Use Keras implementation of ResNet for simple demo
"""

import tensorflow as tf

resnet = tf.keras.applications.ResNet50(weights='imagenet')
