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
                        if out[0][1] > 0.75:
                            pr.predict_severity(left_eye_image)
                        else:
                            st.success("No DR detected")

            with col2:
                if right_eye_image is not None:
                    out = pr.predict_dr(right_eye_image)
                    if out is not None:
                        if out[0][1] > 0.75:
                            pr.predict_severity(right_eye_image)
                        else:
                            st.success("No DR detected")

def home():
        st.header('What is Diabetic Retinopathy?')
        st.write(
            "Diabetic retinopathy is an eye disease that affects people with diabetes. It damages the blood vessels "
            "within the retina, the light-sensitive tissue at the back of the eye. It can cause vision problems and "
            "lead to blindness if left untreated.")

        st.header('Types of Diabetic Retinopathy')
        st.write(
            "There are two main types: Non-proliferative diabetic retinopathy (NPDR) and Proliferative diabetic "
            "retinopathy (PDR). NPDR is the early stage where blood vessels weaken, leak, or become blocked. PDR is "
            "an advanced stage where new, fragile blood vessels grow in the retina.")

        st.header('Symptoms and Prevention')
        st.write(
            "In the early stages, there might be no noticeable symptoms. As it progresses, symptoms may include blurred "
            "vision, floaters, impaired color vision, or vision loss. Regular eye check-ups, controlling blood sugar, "
            "blood pressure, and leading a healthy lifestyle are essential in preventing and managing diabetic retinopathy.")
