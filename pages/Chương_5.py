import cv2
import numpy as np
import streamlit as st
from PIL import Image
L = 256
#-----Function Chapter 5-----#
def CreateMotionfilter(M, N):
    H = np.zeros((M,N), np.complex128)
    a = 0.1
    b = 0.1
    T = 1
    for u in range(0, M):
        for v in range(0, N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            if np.abs(phi) < 1.0e-6:
                RE = T*np.cos(phi)
                IM = -T*np.sin(phi)
            else:
                RE = T*np.sin(phi)/phi*np.cos(phi)
                IM = -T*np.sin(phi)/phi*np.sin(phi)
            H.real[u,v] = RE
            H.imag[u,v] = IM
    return H

def CreateMotionNoise(imgin):
    M, N = imgin.shape
    f = imgin.astype(np.float64)
    # Buoc 1: DFT
    F = np.fft.fft2(f)
    # Buoc 2: Shift vao the center of the image
    F = np.fft.fftshift(F)

    # Buoc 3: Tao bo loc H
    H = CreateMotionfilter(M, N)

    # Buoc 4: Nhan F voi H
    G = F*H

    # Buoc 5: Shift return
    G = np.fft.ifftshift(G)

    # Buoc 6: IDFT
    g = np.fft.ifft2(G)
    g = g.real
    g = np.clip(g, 0, L-1)
    g = g.astype(np.uint8)
    return g

def CreateInverseMotionfilter(M, N):
    H = np.zeros((M,N), np.complex128)
    a = 0.1
    b = 0.1
    T = 1
    phi_prev = 0
    for u in range(0, M):
        for v in range(0, N):
            phi = np.pi*((u-M//2)*a + (v-N//2)*b)
            if np.abs(phi) < 1.0e-6:
                RE = np.cos(phi)/T
                IM = np.sin(phi)/T
            else:
                if np.abs(np.sin(phi)) < 1.0e-6:
                    phi = phi_prev
                RE = phi/(T*np.sin(phi))*np.cos(phi)
                IM = phi/(T*np.sin(phi))*np.sin(phi)
            H.real[u,v] = RE
            H.imag[u,v] = IM
            phi_prev = phi
    return H

def DenoiseMotion(imgin):
    M, N = imgin.shape
    f = imgin.astype(np.float64)
    # Buoc 1: DFT
    F = np.fft.fft2(f)
    # Buoc 2: Shift vao the center of the image
    F = np.fft.fftshift(F)

    # Buoc 3: Tao bo loc H
    H = CreateInverseMotionfilter(M, N)

    # Buoc 4: Nhan F voi H
    G = F*H

    # Buoc 5: Shift return
    G = np.fft.ifftshift(G)

    # Buoc 6: IDFT
    g = np.fft.ifft2(G)
    g = g.real
    g = np.clip(g, 0, L-1)
    g = g.astype(np.uint8)
    return g

# ========================== STREAMLIT UI ==========================

st.title("Xử lý ảnh chương 5")

uploaded_file = st.file_uploader("Chọn ảnh", type=["png", "jpg", "jpeg", "tif"])

if uploaded_file:
    # Đọc ảnh gốc
    img = Image.open(uploaded_file).convert("L")
    img = img.resize((256, 256)) 
    img_np = np.array(img)

    st.image(img, caption="Ảnh gốc", use_container_width=True)

    # ComboBox chọn chức năng
    option = st.selectbox(
        "Chọn chức năng xử lý:",
        (
            "Làm mờ ảnh",
            "Lọc ảnh nhiễu"
        )
    )

    # Xử lý theo lựa chọn
    if option == "Tạo nhiễu ảnh":
        result = CreateMotionNoise(img_np)
    elif option == "Lọc nhiều nhiễu ảnh":
        result = DenoiseMotion(img_np)
    else:
        result = img_np

    st.image(result, caption="Kết quả xử lý", use_container_width=True)