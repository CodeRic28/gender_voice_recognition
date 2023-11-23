import tensorflow as tf
import keras
import os
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
from spectrogram import create_spectrogram

print(os.getcwd())

load_model = tf.keras.models.load_model('model_2.h5')

base_dir = 'images'
training_dir = os.path.join(base_dir,"training")
validation_dir = os.path.join(base_dir,"validation")

# Directory with training male female spectrograms
train_male_dir = os.path.join(training_dir,'male')
train_female_dir = os.path.join(training_dir,'female')

# Directory with validation male female spectrograms
valid_male_dir = os.path.join(validation_dir,'male')
valid_female_dir = os.path.join(validation_dir,'female')

# Load the spectrogram image you want to predict
# spectrogram_image_path = 'SAVING/sample-002951_spectrogram.png'
path ='images/training/female'
files = os.listdir(path)
# for file in files:
#     spectrogram_image_path = os.path.join(path,file)
#     img = load_img(spectrogram_image_path, target_size=(150, 150))
#     x = img_to_array(img)
#     x = x / 255.0  # Normalize the image data if it's not already
#     x = np.expand_dims(x, axis=0)  # Add a batch dimension
#
#     # Make the prediction
#     predictions = load_model.predict(x)
#     print(predictions)
#     # Interpret the prediction
#     if predictions[0][0] > predictions[0][1]:
#         print("Female")
#     else:
#         print("Male")

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
