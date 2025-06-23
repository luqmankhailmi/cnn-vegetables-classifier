# source: https://keras.io/examples/vision/image_classification_from_scratch/

import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt

image_size = (100, 100)
batch_size = 16

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "data",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
    # label_mode='categorical' is added
    label_mode='categorical'
)

# These two lines below are added
class_names = train_ds.class_names
print("Classes:", class_names)

data_augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def data_augmentation(images):
    for layer in data_augmentation_layers:
        images = layer(images)
    return images

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        units = 1
    else:
        units = num_classes

    x = layers.Dropout(0.25)(x)
    # We specify activation=None so as to return logits
    outputs = layers.Dense(units, activation=None)(x)
    return keras.Model(inputs, outputs)

# num_classes changed from 2 to 5
model = make_model(input_shape=image_size + (3,), num_classes=5)

epochs = 100

callbacks = [
    keras.callbacks.ModelCheckpoint("save_at_{epoch}.keras"),
]
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    # Binary is changed to Categorical
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=[keras.metrics.CategoricalAccuracy(name="acc")],
)
model.fit(
    train_ds,
    epochs=epochs,
    callbacks=callbacks,
    validation_data=val_ds,
)

model.save("fruit_classifier_model.h5")

# Binary classification code: need to remove (commented)
#img = keras.utils.load_img("data\\GinsengJawa\\IMG_20190206_104532.jpg", target_size=image_size)
#img_array = keras.utils.img_to_array(img)
#img_array = keras.ops.expand_dims(img_array, 0)  # Create batch axis

#predictions = model.predict(img_array)
#score = float(keras.ops.sigmoid(predictions[0][0]))
#print(f"This image is {100 * (1 - score):.2f}% Ginseng Jawa and {100 * score:.2f}% Jerangau.")

# New code
# Load image
img = keras.utils.load_img("data\\Apple\\Apple Ee02441.png", target_size=image_size)  # <-- replace with your image path
img_array = keras.utils.img_to_array(img)
img_array = keras.ops.expand_dims(img_array, 0)  # Add batch dimension

# Predict
predictions = model.predict(img_array)
probabilities = keras.ops.softmax(predictions[0])  # Convert logits to probabilities

# Get result
predicted_index = np.argmax(probabilities)
predicted_class = class_names[predicted_index]
confidence = float(probabilities[predicted_index]) * 100

# Output prediction
print(f"Predicted: {predicted_class} ({confidence:.2f}%)")

# Optional: print all class probabilities
for i, class_name in enumerate(class_names):
    print(f"{class_name}: {probabilities[i]*100:.2f}%")
