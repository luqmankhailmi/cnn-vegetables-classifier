import tensorflow as tf
import keras
from keras.models import  load_model
import streamlit as st
import numpy as np
from PIL import Image

st.header('What is this vegetable?')
model = load_model('Vegetable_Classifier_Model.keras')
data_cat = ['cabbage',
 'carrot',
 'cauliflower',
 'eggplant',
 'halia']
img_height = 180
img_width = 180
# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess image
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((img_width, img_height))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_batch = tf.expand_dims(img_array, 0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_batch)
    score = tf.nn.softmax(predictions[0])

    # Display image and results
    st.image(image, width=200)
    st.write('Vegetable in image is: **{}**'.format(data_cat[np.argmax(score)]))
    st.write('With accuracy of: **{:.2f}%**'.format(100 * np.max(score)))
