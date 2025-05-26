import streamlit as st
from PIL import Image
from pathlib import Path
import base64
def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_base64 = get_base64_image("background.jpg")
logo_path = "hcmute.png"  # đổi tên nếu ảnh bạn khác
logo_base64 = get_base64_image(logo_path)
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: right 20% center;
        background-attachment: fixed;
    }}
    </style>
    
""", unsafe_allow_html=True)
st.markdown(f"""
    <style>
    .logo-container {{
        position: fixed;
        top: 70px;       
        right: 30px;     
        z-index: 100;
    }}
    .logo-container img {{
        height: 200px;
    }}
    </style>
    <div class="logo-container">
        <img src="data:image/png;base64,{logo_base64}">
    </div>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    .welcome-box {
        position: fixed;
        top: 20%;
        left: 50%;
        transform: translate(-50%, -50%);
        background-color: #ffffffcc;
        padding: 30px 50px;
        border-radius: 15px;
        box-shadow: 0 0 20px rgba(0,0,0,0.2);
        z-index: 999;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }
    </style>
    <div class="welcome-box">
         Chào mừng bạn đến với Website Xử Lý Ảnh<br>
    </div>
""", unsafe_allow_html=True)

