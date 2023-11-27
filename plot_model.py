import matplotlib.pyplot as plt
import matplotlib.patches as patches

def add_layer(ax, name, shape, position):
    ax.add_patch(patches.Rectangle(position, shape[0], shape[1], fill=None, edgecolor='black'))
    ax.text(position[0] + shape[0]/2, position[1] + shape[1]/2, name, ha='center', va='center')

fig, ax = plt.subplots(figsize=(8, 8))

# Input layer
add_layer(ax, 'Input\n(150, 150, 3)', (1, 1), (0, 0))

# Convolutional layer 1
add_layer(ax, 'Conv2D\n(16, 3x3)\nReLU', (2, 2), (1, 0))

# MaxPooling layer 1
add_layer(ax, 'MaxPooling\n(2x2)', (1, 1), (3, 0))

# Convolutional layer 2
add_layer(ax, 'Conv2D\n(32, 3x3)\nReLU', (2, 2), (4, 0))

# MaxPooling layer 2
add_layer(ax, 'MaxPooling\n(2x2)', (1, 1), (6, 0))

# Convolutional layer 3
add_layer(ax, 'Conv2D\n(64, 3x3)\nReLU', (2, 2), (7, 0))

# MaxPooling layer 3
add_layer(ax, 'MaxPooling\n(2x2)', (1, 1), (9, 0))

# Flatten layer
add_layer(ax, 'Flatten', (1, 1), (10, 0))

# Dense layer 1
add_layer(ax, 'Dense\n(512)\nReLU', (2, 2), (11, 0))

# Dense layer 2
add_layer(ax, 'Dense\n(2)\nSoftmax', (2, 2), (13, 0))

# Output layer
add_layer(ax, 'Output\n(2 classes)', (1, 1), (15, 0))

ax.set_xlim(0, 16)
ax.set_ylim(0, 2)

# Remove axes
ax.set_xticks([])
ax.set_yticks([])

plt.show()

# ______________________________________________________________________________

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import plot_model
import pydot

# Define the CNN model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(2, activation='softmax')
])

# Visualize the model
plot_model(model, to_file='cnn_model.png', show_shapes=True, show_layer_names=False)