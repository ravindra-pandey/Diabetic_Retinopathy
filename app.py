import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import os
st.set_page_config(page_icon="hello",layout="wide")

uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
st.write((os.listdir("./models")))
model = tf.keras.models.load_model("models/binary_model.h5")
augmentor=pickle.load(open("serialized_files/augmentor.pkl","rb"))

level=st.slider(label="depth of prediction",min_value=1, max_value=15)

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
    augmented_images=np.array([augmentor(image=image_array)["image"] for i in range(level)])        
    out = model.predict(augmented_images)
    out=np.sum(out,axis=0)/level
    col1,col2=st.columns(2)
    with col1:
        st.image(image, caption="Uploaded Image",width=400)
    with col2:
        bar2 = st.progress(0, text=f"Infected : {out[1] * 100:.2f}%")

        for i in range(int(out[1] * 100)):
            bar2.progress(i + 1, text=f"chances of infection : {out[1] * 100:.2f}%")
