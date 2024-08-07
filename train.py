import os
import warnings
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import ModelCheckpoint

warnings.filterwarnings('ignore')

def limit_data(data_dir):
    a = []
    for i in os.listdir(data_dir):
        for j in os.listdir(os.path.join(data_dir, i)):
            a.append((os.path.join(data_dir, i, j), i))
    return pd.DataFrame(a, columns=['filename', 'class'])

# Load and prepare data
general_path = r"try/data/images"
path_train_images = os.path.join(general_path, "train")
path_test_images = os.path.join(general_path, "validation")

limited_train_data = limit_data(path_train_images)
limited_test_data = limit_data(path_test_images)

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_dataset = datagen.flow_from_dataframe(dataframe=limited_train_data, batch_size=16)
test_dataset = datagen.flow_from_dataframe(dataframe=limited_test_data, batch_size=16)

# Create and train CNN model
non_trainable_model = Xception(
    weights='imagenet',
    include_top=False,
    input_shape=(train_dataset.image_shape[0], train_dataset.image_shape[1], train_dataset.image_shape[2])
)

for layer in non_trainable_model.layers:
    layer.trainable = False

x = non_trainable_model.output
x = tf.keras.layers.Conv2D(filters=50, kernel_size=(3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
flatten = tf.keras.layers.Flatten()(x)
predictions = tf.keras.layers.Dense(7, activation='softmax')(flatten)

model = tf.keras.Model(inputs=non_trainable_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Add checkpoint callback
checkpoint = ModelCheckpoint('emotion_model_checkpoint.keras', save_best_only=True, monitor='val_loss', mode='min')

model.fit(train_dataset, epochs=10, validation_data=test_dataset, callbacks=[checkpoint])

# Save the final model
model.save('emotion_model.keras')

# Extract features
model2 = tf.keras.Model(inputs=non_trainable_model.input, outputs=flatten)
train_data_flatten = model2.predict(train_dataset)