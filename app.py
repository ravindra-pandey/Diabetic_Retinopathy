import streamlit as st
from streamlit_option_menu import option_menu
import app_sections as pr

st.set_page_config(page_title="Diabetic Retinopathy", page_icon="👁️", layout="wide")

st.header(
    ":GREY[Diabetic Retinopathy detection]",
    divider="grey",
    anchor="diabetic_retinopathy",
)
with st.container():  
    selected = option_menu(
        menu_title=None,
        options=["Home","Test DR", "Samples"],
        icons=["house","eye", ""],
        orientation="horizontal",
    )
if selected =="Home":
    pr.home()
elif selected =="Test DR":
    pr.prediction_section()
