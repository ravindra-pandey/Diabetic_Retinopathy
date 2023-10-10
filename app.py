import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
model = tf.keras.models.load_model("model.h5")

if uploaded_file is not None:
    st.write("You uploaded:", uploaded_file.name)
    image = Image.open(uploaded_file)
    width, height = image.size

    # Define cropping boundaries
    left = int(width * 0.05)
    right = int(width * 0.95)
    top = int(height * 0.08)
    bottom = int(height * 0.92)

    # Crop the imagestre
    image = image.crop((left, top, right, bottom))
    
    # Resize the cropped image
    image = image.resize((224, 224))
    # Convert the PIL image to a numpy array
    image_array = np.array(image)
    st.write(image_array.shape)

    # Predict using the model
    out = model.predict(image_array.reshape(-1, 224, 224, 3))

    st.image(image, caption="Uploaded Image", use_column_width=True)
    bar1 = st.progress(0, text="Normal")
    bar2 = st.progress(0, text="Infected")

    for i in range(int(out[0][0] * 100)):
        bar1.progress(i + 1, text="Normal")

    for i in range(int(out[0][1] * 100)):
        bar2.progress(i + 1, text="Infected")
