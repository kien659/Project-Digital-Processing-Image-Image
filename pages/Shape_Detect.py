import cv2
import numpy as np
from PIL import Image
import streamlit as st
import pandas as pd
def is_rectangle_or_square(approx):
    def angle(p1, p2, p3):
        a = np.array(p1)
        b = np.array(p2)
        c = np.array(p3)
        ab = a - b
        cb = c - b
        cos_angle = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        return np.degrees(angle)

    if len(approx) != 4:
        return False

    angles = []
    for i in range(4):
        p1 = approx[i][0]
        p2 = approx[(i + 1) % 4][0]
        p3 = approx[(i + 2) % 4][0]
        ang = angle(p1, p2, p3)
        angles.append(ang)

    return all(80 <= a <= 100 for a in angles)  # Góc gần vuông
def phan_nguong(imgin):
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)
    for x in range(M):
        for y in range(N):
            r = imgin[x, y]
            imgout[x, y] = 255 if r == 63 else 0
    imgout = cv2.medianBlur(imgout, 7)
    return imgout

def du_doan_hinh_dang(imgin):
    gray = cv2.cvtColor(imgin, cv2.COLOR_RGB2GRAY)
    temp = phan_nguong(gray)

    contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imgout = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    shape_info_list = []

    for i, cnt in enumerate(contours):
        #điều chỉnh độ chính xác 4%
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        #Làm trơn contour và chuyển nó thành polygon (đa giác) có số cạnh xác định
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        #Đếm số cạnh
        num_sides = len(approx)
        x, y, w, h = cv2.boundingRect(approx)
        perimeter = cv2.arcLength(cnt, True)
        area = cv2.contourArea(cnt)

        # Tính circularity (hình tròn)
        if perimeter == 0:  # Tránh chia cho 0
            circularity = 0
        else:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        shape = "Unknown"  # Gán giá trị mặc định cho shape
        # Phân loại dựa trên số cạnh
        if num_sides == 3:
            shape = "Hinh Tam Giac"
        elif num_sides == 4:
            if is_rectangle_or_square(approx):
                aspect_ratio = w / float(h)
                shape = "Hinh Vuong" if 0.95 <= aspect_ratio <= 1.05 else "Hinh Chu Nhat"
        elif num_sides > 4:
            if circularity > 0.8:  # Nếu circularity gần 1, xác định là hình tròn
                aspect_ratio = w / float(h)
                if 0.9 <= aspect_ratio <= 1.1:
                    shape = "Hinh Tron"
        else:
            shape = "Unknown"
        #Tính tọa độ tâm
        cx = int(x + w / 2)
        cy = int(y + h / 2)

        # Vẽ shape và nhãn
        cv2.drawContours(imgout, [cnt], -1, (0, 255, 0), 2)
        cv2.putText(imgout, shape, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


    return imgout, temp, shape_info_list
st.title("📐 Nhận diện Shape")

uploaded_file = st.file_uploader("Tải ảnh (có 1 hoặc nhiều shape)", type=["png", "jpg", "jpeg", "bmp"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    img_out, thresh_img, shape_list = du_doan_hinh_dang(img_np)
    st.image(image, caption="Ảnh đã tải", use_container_width=True)
    st.subheader("Ảnh sau phân ngưỡng")
    st.image(thresh_img, clamp=True, channels="GRAY", use_container_width=True)

    st.subheader("Kết quả nhận dạng")
    st.image(cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB), use_container_width=True)

    if not shape_list:
        st.warning("Không nhận diện được hình nào.")