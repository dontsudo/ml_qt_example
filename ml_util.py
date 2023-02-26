import tensorflow as tf
import numpy as np


def load_model(fname):
    return tf.keras.models.load_model(fname)


def predict(model, image):
    mnist_np = np.array(image)
    predicted = model.predict(np.array([mnist_np]))

    return np.argmax(predicted)
