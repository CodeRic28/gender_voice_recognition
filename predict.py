import tensorflow as tf
import keras
import os
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

print(os.getcwd())

load_model = tf.keras.models.load_model('model.h5')

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
# spectrogram_image_path = 'images/validation/female/sample-000362.png'
# spectrogram_image_path = 'SAVING/sample-000006_spectrogram.png'
# spectrogram_image_path = 'SAVING/sample-000014_spectrogram.png'
# spectrogram_image_path = 'SAVING/sample-000428_spectrogram.png'
# spectrogram_image_path = 'SAVING/sample-002014_spectrogram.png'
# spectrogram_image_path = 'SAVING/sample-002428_spectrogram.png'
# spectrogram_image_path = 'SAVING/sample-002951_spectrogram.png'
# spectrogram_image_path = 'images/training/female/sample-000088.png'
path ='images/training/female'
files = os.listdir(path)
for file in files:
    spectrogram_image_path = os.path.join(path,file)
    img = load_img(spectrogram_image_path, target_size=(150, 150))
    x = img_to_array(img)
    x = x / 255.0  # Normalize the image data if it's not already
    x = np.expand_dims(x, axis=0)  # Add a batch dimension

    # Make the prediction
    predictions = load_model.predict(x)

    # Interpret the prediction
    if predictions[0] > 0.52:
        print("Predicted as male")
    else:
        print("Predicted as female")

# img = load_img(spectrogram_image_path, target_size=(150, 150))
# x = img_to_array(img)
# x = x / 255.0  # Normalize the image data if it's not already
# x = np.expand_dims(x, axis=0)  # Add a batch dimension
#
# # Make the prediction
# predictions = load_model.predict(x)
#
# # Interpret the prediction
# if predictions[0] > 0.5:
#     print("Predicted as male")
# else:
#     print("Predicted as female")
