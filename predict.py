import tensorflow as tf
import keras
import os
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from spectrogram import create_spectrogram


load_model = tf.keras.models.load_model('model.h5')


img = load_img(spectrogram_image_path, target_size=(150, 150))
x = img_to_array(img)
x = x / 255.0  # Normalize the image data if it's not already
x = np.expand_dims(x, axis=0)  # Add a batch dimension

# Make the prediction
predictions = load_model.predict(x)

if predictions[0][0] > predictions[0][1]:
    print("Female")
else:
    print("Male")
