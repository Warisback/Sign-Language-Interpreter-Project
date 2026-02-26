import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

SIGNS = ["hello", "yes", "no", "help", "please", "thankyou"]
IMG_SIZE = 224

# ── DATA LOADING ─────────────────────────────────────────────────────────────
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

train_data = datagen.flow_from_directory(
    "data/",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16,
    class_mode="categorical",
    subset="training"
)

val_data = datagen.flow_from_directory(
    "data/",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=16,
    class_mode="categorical",
    subset="validation"
)

print("Classes:", train_data.class_indices)

# ── BUILD MODEL ───────────────────────────────────────────────────────────────
base = MobileNetV2(weights="imagenet", include_top=False,
                   input_shape=(IMG_SIZE, IMG_SIZE, 3))
base.trainable = False  # freeze pretrained weights

x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(len(SIGNS), activation="softmax")(x)

model = Model(inputs=base.input, outputs=output)
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# ── TRAIN ─────────────────────────────────────────────────────────────────────
print("Training...")
model.fit(train_data, validation_data=val_data, epochs=10)

# ── SAVE ──────────────────────────────────────────────────────────────────────
model.save("signbridge_model.h5")

# save class labels
import json
labels = {v: k for k, v in train_data.class_indices.items()}
with open("labels.json", "w") as f:
    json.dump(labels, f)

print("Model saved!")
print("Labels:", labels)