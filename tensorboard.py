import tensorflow as tf
from tensorflow import keras
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import numpy as np


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images[:500]
train_labels = train_labels[:500]

test_images = test_images[:500]
test_labels = test_labels[:500]

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128 , activation=tf.nn.relu),
    keras.layers.Dense(10 , activation = tf.nn.softmax)
])

tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

model.compile(optimizer=tf.train.AdamOptimizer(),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(train_images , train_labels, epochs = 3 , callbacks=[tensorboard])

test_loss, test_acc = model.evaluate(test_images, test_labels)

print("The Test Accuracy is :", test_acc)
