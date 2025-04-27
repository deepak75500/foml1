
import tensorflow as tf
import opendatasets as od
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
import matplotlib.pyplot as plt
import os
od.download('https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset')
base_dir = r"C:\Users\deepak\brain-tumor-mri-dataset"
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    base_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=5
)

model.save('brain_tumor_detector.h5')
model = tf.keras.models.load_model('brain_tumor_detector.h5')

test_images_dir = r"C:\Users\deepak\brain-tumor-mri-dataset\Testing\pituitary"  
test_images = os.listdir(test_images_dir)[:5]  

for img_name in test_images:
    img_path = os.path.join(test_images_dir, img_name)
    
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    prediction = model.predict(img_array)
    result = "Tumor detected 😔" if prediction[0][0] > 0.5 else "No tumor detected 🙂"
    plt.imshow(img)
    plt.axis('off')
    plt.title(result)
    plt.show()
