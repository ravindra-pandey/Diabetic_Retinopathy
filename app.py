import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

uploaded_file = st.file_uploader("Choose a file", type=["jpg","jpeg","png"])
model=tf.keras.models.load_model("model.h5")

if uploaded_file is not None:
    st.write("You uploaded:", uploaded_file.name)
    image=np.array(Image.open(uploaded_file))
    image=cv2.resize(image[int(image.shape[0]*0.2):int(image.shape[0]*0.8),int(image.shape[1]*0.15):int(image.shape[1]*0.85)],(224,224))
    out=model.predict(image.reshape(-1,224,224,3))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    bar1=st.progress(0,text="Normal")
    bar2=st.progress(0,text="Infected")
    for i in range(int(out[0][0]*100)):
        bar1.progress(i+1,text="Normal")
    for i in range(int(out[0][1]*100)):
        bar2.progress(i+1,text="Infected")