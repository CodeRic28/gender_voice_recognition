import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

base_dir = 'images'

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
    # tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    # tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')
])

print(model.summary())

algo = Adam(lr=0.0001)
model.compile(
    optimizer=algo,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.save("model_lr.h5")


# All images will be rescaled by 1./255.
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

# --------------------
# Flow training images in batches of 20 using train_datagen generator
# --------------------
train_generator = train_datagen.flow_from_directory(training_dir,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

# --------------------
# Flow validation images in batches of 20 using test_datagen generator
# --------------------
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                         batch_size=20,
                                                         class_mode  = 'binary',
                                                         target_size = (150, 150))

history = model.fit(
            train_generator,
            epochs=10,
            validation_data=validation_generator,
            verbose=1
)


loss, accuracy = model.evaluate(validation_generator)



