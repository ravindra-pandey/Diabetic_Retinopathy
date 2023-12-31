import pickle
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st

retina_detector = tf.keras.models.load_model("models/retina_detection.h5")
model = tf.keras.models.load_model("models/binary_model.h5")
severity_model = tf.keras.models.load_model("models/severity_model.h5")
encoder = pickle.load(open("serialized_files/label_encoder.pkl", "rb"))


def preprocess_image(image):
    # width, height = image.size
    # left = int(width * 0.05)
    # right = int(width * 0.95)
    # top = int(height * 0.08)
    # bottom = int(height * 0.92)

    # image = image.crop((left, top, right, bottom))
    image = image.resize((224, 224))
    image = np.array(image)
    return image


def read_image(uploaded_image):
    IMAGE = Image.open(uploaded_image).convert("RGB")
    image = preprocess_image(IMAGE)
    return IMAGE, image


def predict_dr(uploaded_image,show_im=True):
    IMAGE, image = read_image(uploaded_image)
    if show_im:
        st.image(image)
    # temp_img = Image.fromarray(np.pad(IMAGE, 1).astype("uint8"), "RGB")
    retina_score = retina_detector.predict(np.array([np.array(IMAGE.resize((32, 32)))]))

    if retina_score[0][1] > 0.95:
        bar = st.progress(0, text="chance of infection")
        out = model.predict(np.array([image]))
        if out[0][1] > 0.55:
            text_color = "red"
        else:
            text_color = "green"
        for i in range(int(out[0][1] * 100)):
            bar.progress(
                i + 1,
                text=f":{text_color}[chances of infection: {out[0][1] * 100:.2f}%]",
            )
        st.divider()
    else:
        st.error(
            '''It is not an retina image *or* \n It has been zoomed IN unnecessarily Please get the better picture for testing ",
            icon="🚨,
        ''')
        out = None
    return out


def predict_severity(uploaded_image):
    _, image = read_image(uploaded_image)

    severity = severity_model.predict(np.array([image]))
    bar2 = st.progress(0, text=encoder.inverse_transform([0])[0])
    bar3 = st.progress(0, text=encoder.inverse_transform([1])[0])
    bar4 = st.progress(0, text=encoder.inverse_transform([3])[0])
    bar5 = st.progress(
        0,
        text=encoder.inverse_transform([2])[0],
    )
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
