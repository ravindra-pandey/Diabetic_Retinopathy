import streamlit as st
import tensorflow as tf
import numpy as np

import predictions as pr

st.set_page_config(page_title="Diabetic Retinopathy", page_icon="👁️", layout="wide")

st.header("Diabetic Retinopathy detection")
def dr_prediction():
        
    with st.form("Enter your details here..."):
        col1,col2=st.columns(2)
        with col1:
            name=st.text_input("Name:",max_chars=25)
        with col2:
            age=st.number_input("Age:",max_value=150,step=1)
        col1,col2=st.columns(2)
        with col1:
            left_eye_image=st.file_uploader("Enter your left eye image",type=["jpg","png","jpeg"])
        with col2:
            right_eye_image=st.file_uploader("Enter your right eye image",type=["jpg","png","jpeg"])
        submit=st.form_submit_button("Submit")
    
        if submit:
            col1,col2=st.columns(2)
            with col1:
                if left_eye_image is not None:
                    out=pr.predict_dr(left_eye_image)
                    if out is not None:
                        if out[0][1]>0.75 :
                            pr.predict_severity(left_eye_image)
                        else:
                            st.success("No DR detected")
        
            with col2:
                if right_eye_image is not None:
                    out=pr.predict_dr(right_eye_image)
                    if out is not None:
                        if out[0][1]>0.75 :
                            pr.predict_severity(right_eye_image)
                        else:
                            st.success("No DR detected")
        