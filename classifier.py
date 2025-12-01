import tensorflow as tf
from tensorflow.keras import layers, models
import streamlit as st
from PIL import Image
import numpy as np

train_dir = "dataset/train"
test_dir = "dataset/test"
batch_size = 16
img_size = (64, 64)

train_dataset = tf.keras.utils.image_dataset_from_directory(  #make image size 
    train_dir,
    image_size=img_size,
    batch_size=batch_size
)

test_dataset = tf.keras.utils.image_dataset_from_directory(      #make image size 
    test_dir,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_dataset.class_names     # # Get the names

model = models.Sequential([                  # Define the Sequential Convolutional Neural Network model
    layers.Rescaling(1./255, input_shape=(64,64,3)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_dataset, epochs=10, validation_data=test_dataset)
model.save("custom_cnn_model.h5")

st.title("Custom Image Classifier")
st.write("Hello! Upload an image below")

uploaded_file = st.file_uploader("Choose an image (PNG or JPG) to classify", type=["png","jpg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).resize((64,64))
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    with st.spinner("Analyzing the image!!!"):
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_class = class_names[predicted_index]
        confidence = predictions[0][predicted_index] * 100
    
    st.success(f"I think this is a **{predicted_class}** with **{confidence:.2f}%** confidence!")
