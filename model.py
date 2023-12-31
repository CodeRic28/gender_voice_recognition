import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

base_dir = 'images'
print(tf.device('/device:GPU:1'))
print("Contents of base dir: ",os.listdir(base_dir))
print("Contents of train dir: \n", os.listdir(f"{base_dir}/training"))
print("Contents of validation dir: \n", os.listdir(f"{base_dir}/validation"))

training_dir = os.path.join(base_dir,"training")
validation_dir = os.path.join(base_dir,"validation")

# Directory with training male female spectrograms
train_male_dir = os.path.join(training_dir,'male')
train_female_dir = os.path.join(training_dir,'female')

# Directory with validation male female spectrograms
valid_male_dir = os.path.join(validation_dir,'male')
valid_female_dir = os.path.join(validation_dir,'female')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2,activation='softmax')
])

print(model.summary())

# algo = Adam(lr=0.01)
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

# --------------------
# Flow training images in batches of 20 using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_directory(training_dir,
                                                    batch_size=20,
                                                    class_mode='categorical',
                                                    target_size=(150, 150))

# --------------------
# Flow validation images in batches of 20 using test_datagen generator
# --------------------
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode  = 'categorical',
                                                         target_size = (150, 150))

history = model.fit(
            train_generator,
            epochs=10,
            validation_data=validation_generator,
            verbose=1
)

import matplotlib.pyplot as plt
# Plot the model results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.figure()
plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

loss, accuracy = model.evaluate(validation_generator)

model.save("model_2.h5")

