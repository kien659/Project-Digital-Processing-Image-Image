import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import cv2
import numpy as np
# Load model YOLO
@st.cache_resource
def load_model():
    model = YOLO('best.pt')  # file .pt từ Google Colab
    return model

model = load_model()

# Giao diện
st.title("🍉 Nhận diện trái cây bằng YOLOv8")
st.write("Tải ảnh lên để phát hiện các loại trái cây trong ảnh.")

uploaded_file = st.file_uploader("Chọn ảnh", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Ảnh đã tải", use_container_width=True)

    # Chạy mô hình
    with st.spinner("🔍 Đang nhận diện..."):
        results = model.predict(image)

        # Vẽ bounding box
        result_img = results[0].plot()  # -> BGR array (OpenCV)
        result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)  # Chuyển sang RGB

        st.image(result_img_rgb, caption="Kết quả dự đoán", use_container_width=True)

        # Liệt kê kết quả
        st.subheader("📋 Chi tiết đối tượng phát hiện:")
        names = model.names  # Tên lớp
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            st.write(f"- {names[cls_id]} ({conf:.2%})")
            