import streamlit as st
import cv2
import numpy as np
from PIL import Image
L=256
def Spectrum(imgin):
    M,N = imgin.shape
    # Buoc 1 va 2 Mo rong anh co kich thuoc PxQ
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp=np.zeros((P,Q), np.float32)
    fp[:M,:N] = 1.0*imgin/(L-1)
    # Buoc 3 Nhan fp voi(-1)^(x+y)
    for x in range(0,M):
        for y in range(0,N):
            if(x+y) % 2 ==1:
                fp[x,y] = -fp[x,y]
    #Buoc 4 DFT
    F = cv2.dft(fp,flags=cv2.DFT_COMPLEX_OUTPUT)
    #Tinh pho
    FR = F[:,:,0]
    FI = F[:,:,1]
    S = np.sqrt(FR**2 + FI**2)
    S = np.clip(S,0,L-1)
    imgout = S.astype(np.uint8)
    return imgout


def CreateNotchFilter(P,Q):
    H=np.ones((P,Q,2),np.float32)
    H[:,:,1] = 0.0
    u1,v1 = 45,58
    u2,v2 = 86,58
    u3,v3 = 40,119
    u4,v4 = 82,119

    u5,v5 = P-45,Q-58
    u6,v6 = P-86,Q-58
    u7,v7 = P-40,Q-119
    u8,v8 = P-82,Q-119

    D0 = 15
    for u in range(0,P):
        for v in range(0,Q):
            #u1,v1
            D = np.sqrt((1.0*u-u1)**2+(1.0*v-v1)**2)
            if D <= D0 :
                H[u,v,0]=0.0
                
             #u2,v2
            D = np.sqrt((1.0*u-u2)**2+(1.0*v-v2)**2)
            if D <= D0 :
                H[u,v,0]=0.0

             #u3,v3
            D = np.sqrt((1.0*u-u3)**2+(1.0*v-v3)**2)
            if D <= D0 :
                H[u,v,0]=0.0
             #u4,v4
            D = np.sqrt((1.0*u-u4)**2+(1.0*v-v4)**2)
            if D <= D0 :
                H[u,v,0]=0.0
             #u5,v5
            D = np.sqrt((1.0*u-u5)**2+(1.0*v-v5)**2)
            if D <= D0 :
                H[u,v,0]=0.0
             #u6,v6
            D = np.sqrt((1.0*u-u6)**2+(1.0*v-v6)**2)
            if D <= D0 :
                H[u,v,0]=0.0
             #u7,v7
            D = np.sqrt((1.0*u-u7)**2+(1.0*v-v7)**2)
            if D <= D0 :
                H[u,v,0]=0.0
             #u8,v8
            D = np.sqrt((1.0*u-u8)**2+(1.0*v-v8)**2)
            if D <= D0 :
                H[u,v,0]=0.0
    return H
def RemoveMoire(imgin):
    M,N = imgin.shape
    # Buoc 1 va 2 Mo rong anh co kich thuoc PxQ
    P = cv2.getOptimalDFTSize(M)
    Q = cv2.getOptimalDFTSize(N)
    fp=np.zeros((P,Q), np.float32)
    fp[:M,:N] = 1.0*imgin
    # Buoc 3 Nhan fp voi(-1)^(x+y)
    for x in range(0,M):
        for y in range(0,N):
            if(x+y) % 2 == 1:
                fp[x,y] = -fp[x,y]
    #Buoc 4 DFT
    F = cv2.dft(fp,flags=cv2.DFT_COMPLEX_OUTPUT)
    #Buoc 5 tao bo loc H
    H = CreateNotchFilter(P,Q)
    #Buoc 6
    G = cv2.mulSpectrums(F,H,flags=cv2.DFT_ROWS)
    #Buoc 7 IDFt
    g = cv2.idft(G,flags=cv2.DFT_SCALE)
    #Buoc 8 lay phan thuc co kich thuoc thuc la doi xung qua tam P//2 Q//2
    gR = g[:M,:N,0]
    for x in range(0,M):
        for y in range(0,N):
            if(x+y) % 2 == 1:
                gR[x,y] = -gR[x,y]
    gR = np.clip(gR,0,L-1)
    imgout = gR.astype(np.uint8)

    return imgout
#11/4/2025
def FrequencyFilterimg(imgin, H):
    M,N = imgin.shape
    f = imgin.astype(np.float64)
   
    F = np.fft.fft2(f)
    
    F = np.fft.fftshift(F)
   
    G = F*H
    G = np.fft.fftshift(G)

    g = np.fft.ifft2(G)
    gR = g.real.copy()
    gR = np.clip(gR,0,L-1)
    imgout =  gR.astype(np.uint8)
    return imgout
def Spec(imgin):
    M,N = imgin.shape
    f = imgin.astype(np.float32)/(L-1)
    F = np.fft.fft2(f)
    F = np.fft.fftshift(F)
    S = np.sqrt(F.real**2 + F.imag**2)
    S = np.clip(S,0,L-1)

    imgout = S.astype(np.uint8)
    return imgout
    
def CreateNotchFilterFreq(M,N):
    H = np.ones((M,N),np.complex64)
    H.imag= 0.0
    u1, v1 = 44,55
    u2,v2 = 85,55
    u3,v3 = 40,111
    u4,v4 = 84,111
    
    u5,v5 = M-44,N-55
    u6,v6 = M-85,N-55
    u7,v7 = M-40,N-111
    u8,v8 = M-84,N-111
    D0 = 15
    for u in range(M):
        for v in range(N):
            # u1,v1
            D = np.sqrt((1.0*u-u1)**2 + (1.0*v-v1)**2)
            if D < D0:
                H.real[u,v] = 0.0
            
            # u2,v2
            D = np.sqrt((1.0*u-u2)**2 + (1.0*v-v2)**2)
            if D < D0:
                H.real[u,v] = 0.0
            
            # u3,v3
            D = np.sqrt((1.0*u-u3)**2 + (1.0*v-v3)**2)
            if D < D0:
                H.real[u,v] = 0.0
            
            # u4,v4
            D = np.sqrt((1.0*u-u4)**2 + (1.0*v-v4)**2)
            if D < D0:
                H.real[u,v] = 0.0
            
            # u5,v5
            D = np.sqrt((1.0*u-u5)**2 + (1.0*v-v5)**2)
            if D < D0:
                H.real[u,v] = 0.0
            
            # u6,v6
            D = np.sqrt((1.0*u-u6)**2 + (1.0*v-v6)**2)
            if D < D0:
                H.real[u,v] = 0.0
            
            # u7,v7
            D = np.sqrt((1.0*u-u7)**2 + (1.0*v-v7)**2)
            if D < D0:
                H.real[u,v] = 0.0
            
            # u8,v8
            D = np.sqrt((1.0*u-u8)**2 + (1.0*v-v8)**2)
            if D < D0:
                H.real[u,v] = 0.0
    return H

def RemoveMoireFreq(imgin):
    M,N = imgin.shape
    H = CreateNotchFilterFreq(M,N)
    imgout = FrequencyFilterimg(imgin,H)
    return imgout


#================================ lam mo
def CreateNotchInferenceFilter(M,N):
    H = np.ones((M,N),np.complex64)
    H.imag = 0.0
    D0 = 7
    D1 = 7
    for u in range(0,M):
        for v in range(0,N):
            if u not in range(M//2-D1,M//2+D1+1):
                if v in range(N//2-D0,N//2+D0+1):
                    H.real[u,v]=0.0
    return H
def RemoveInference(imgin):
    M,N = imgin.shape
    H = CreateNotchInferenceFilter(M,N)
    imgout = FrequencyFilterimg(imgin,H)
    return imgout




# ========================== STREAMLIT UI ==========================

st.title("Xử lý ảnh chương 4")

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
            "Xem phổ ảnh (Spectrum)",
            "Loại bỏ moiré (RemoveMoire - spatial)",
            "Lọc tần số moiré (RemoveMoireFreq - frequency)",
            "Loại bỏ nhiễu đường kẻ (RemoveInference)"
        )
    )

    # Xử lý theo lựa chọn
    if option == "Xem phổ ảnh (Spectrum)":
        result = Spectrum(img_np)
    elif option == "Loại bỏ moiré (RemoveMoire - spatial)":
        result = RemoveMoire(img_np)
    elif option == "Loại bỏ moiré (RemoveMoireFreq - frequency)":
        result = RemoveMoireFreq(img_np)
    elif option == "Loại bỏ nhiễu đường kẻ (RemoveInference)":
        result = RemoveInference(img_np)
    else:
        result = img_np

    st.image(result, caption="Kết quả xử lý", use_container_width=True)
