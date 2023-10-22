import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import os

st.set_page_config(page_title="Diabetic Retinopathy", page_icon="👁️", layout="wide")

st.header("Diabetic Retinopathy detection")
uploaded_file = st.file_uploader(
    "Upload the image of retina here....", type=["jpg", "jpeg", "png"]
)

retina_detector=tf.keras.models.load_model("models/retina_detection.h5")
model = tf.keras.models.load_model("models/binary_model.h5")
severity_model = tf.keras.models.load_model("models/severity_model.h5")
encoder = pickle.load(open("serialized_files/label_encoder.pkl", "rb"))

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    width, height = image.size

    left = int(width * 0.05)
    right = int(width * 0.95)
    top = int(height * 0.08)
    bottom = int(height * 0.92)

    image = image.crop((left, top, right, bottom))
    image = image.resize((224, 224))
    image_array = np.array(image)

    retina_score=retina_detector.predict(np.array([np.array(image.resize((32,32)))]))
    predict=False
    if retina_score[0][1]>0.95:
        out = model.predict(np.array([image_array]))
        predict=True
    else:
        st.error("Doesn't look like retina image")

col1, col2 = st.columns([0.55, 0.45])

with col1:
    if uploaded_file is not None:
        st.image(image, caption="Uploaded Image", width=400)
    else:
        st.header("your image will be shown here")
with col2:
    if uploaded_file is not None and predict:
        bar1 = st.progress(0, text=f"Chances of infection")
        for i in range(int(out[0][1] * 100)):
            bar1.progress(i + 1, text=f"chances of infection : {out[0][1] * 100:.2f}%")

        if out[0][1] > 0.3:
            severity = severity_model.predict(np.array([image_array]))
            bar2 = st.progress(0, text=encoder.inverse_transform([0])[0])
            bar3 = st.progress(0, text=encoder.inverse_transform([1])[0])
            bar4 = st.progress(0, text=encoder.inverse_transform([3])[0])
            bar5 = st.progress(0, text=encoder.inverse_transform([2])[0])
            for i in range(int(severity[0][0] * 100)):
                bar2.progress(
                    i + 1,
                    text=f"{encoder.inverse_transform([0])[0]}: {severity[0][0] * 100:.2f}%",
                )

            for i in range(int(severity[0][1] * 100)):
                bar3.progress(
                    i + 1,
                    text=f"{encoder.inverse_transform([1])[0]}: {severity[0][1] * 100:.2f}%",
                )

            for i in range(int(severity[0][3] * 100)):
                bar4.progress(
                    i + 1,
                    text=f"{encoder.inverse_transform([3])[0]}: {severity[0][3] * 100:.2f}%",
                )

            for i in range(int(severity[0][2] * 100)):
                bar5.progress(
                    i + 1,
                    text=f"{encoder.inverse_transform([2])[0]}: {severity[0][2] * 100:.2f}%",
                )
