import os
import streamlit as st
import predictions as pr


def prediction_section():
    with st.form("Enter your details here..."):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name:", max_chars=25)
        with col2:
            age = st.number_input("Age:", max_value=150, step=1)
        col1, col2 = st.columns(2)
        with col1:
            left_eye_image = st.file_uploader(
                "Enter your left eye image", type=["jpg", "png", "jpeg"]
            )
        with col2:
            right_eye_image = st.file_uploader(
                "Enter your right eye image", type=["jpg", "png", "jpeg"]
            )
        submit = st.form_submit_button("Submit")

        if submit:
            col1, col2 = st.columns(2)
            with col1:
                if left_eye_image is not None:
                    out = pr.predict_dr(left_eye_image)
                    if out is not None:
                        if out[0][1] > 0.55:
                            pr.predict_severity(left_eye_image)
                        else:
                            st.success("No DR detected")

            with col2:
                if right_eye_image is not None:
                    out = pr.predict_dr(right_eye_image)
                    if out is not None:
                        if out[0][1] > 0.55:
                            pr.predict_severity(right_eye_image)
                        else:
                            st.success("No DR detected")


def sample_prediction(image_path,key):
    col1, col2 = st.columns(2)
    image_type=image_path.split("/")[-2]
    with col1:
        _, image = pr.read_image(image_path)
        st.image(image, width=300   )
    with col2:
        st.markdown(f'''### This is an {image_type} image Originally. 
                    To test our model click on the button below. ''')
        test = st.button("Test DR",key=key)
        if test == True:
            out = pr.predict_dr(image_path, show_im=False)
            if out is not None:
                if out[0][1] > 0.55:
                    pr.predict_severity(image_path)
                else:
                    st.success("No DR detected")

def sample():
    image_names=[f"{root}/{file}" for root,dir,files in os.walk("samples") for file in files]
    for key,image_name in enumerate(image_names):
        sample_prediction(image_name,key)