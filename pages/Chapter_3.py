import streamlit as st
import cv2
import numpy as np
from PIL import Image

# ===== Định nghĩa các hàm xử lý ảnh ở đây =====
# (Dán toàn bộ các hàm bạn đã cung cấp vào đây)
# Ví dụ:
L=256
def Negative(imgin):
    M,N =imgin.shape
    imgout=np.zeros((M,N),np.uint8) + np.uint8(255)
    for x in range(0,M):
        for y in range(0,N):
            r=imgin[x,y]
            s=L-1-r
            imgout[x,y]=np.uint8(s)
    return imgout
def NegativeColor(imgin):
    #C: Chanel là 3 cho ảnh màu
    M,N,C =imgin.shape
    imgout=np.zeros((M,N,C),np.uint8) 
    for x in range(0,M):
        for y in range(0,N):
            # ảnh màu của opencv là BGR
            # ảnh màu của pillow là RGB
            b=imgin[x,y,0]
            b=L-1-b

            g=imgin[x,y,1]
            g=L-1-g

            r=imgin[x,y,2]
            r=L-1-r

            imgout[x,y,0]=np.uint8(b)
            imgout[x,y,1]=np.uint8(g)
            imgout[x,y,2]=np.uint8(r)
    return imgout 
def Logarit(imgin):
    M,N=imgin.shape
    imgout=np.zeros((M,N),np.uint8)
    c=(L-1.0)/np.log(1.0*L)
    for x in range(0,M):
        for y in range(0,N):
            r=imgin[x,y]
            if r==0:
                r=1
            s=c*np.log(1.0+r)
            imgout[x,y]=np.uint8(s)
    return imgout
def Power(imgin):
    M,N=imgin.shape
    imgout=np.zeros((M,N),np.uint8)
    gamma=5.0
    c=np.power(L-1.0,1-gamma)
    for x in range(0,M):
        for y in range(0,N):
            r=imgin[x,y]
            if r==0:
                r=1
            s=c*np.power(1.0*r,gamma)
            imgout[x,y]=np.uint8(s)
    return imgout 

def PiecewiseLine(imgin):
    L = 256  # Ensure this is defined
    M, N = imgin.shape
    imgout = np.zeros((M, N), np.uint8)

    rmin, rmax = cv2.minMaxLoc(imgin)[0], cv2.minMaxLoc(imgin)[1]
    r1 = rmin
    s1 = 0
    r2 = rmax
    s2 = L - 1

    for x in range(M):
        for y in range(N):
            r = imgin[x, y]

            # Segment I
            if r1 > 0 and r < r1:
                s = (s1 / r1) * r
            # Segment II
            elif r1 != r2 and r < r2:
                s = ((s2 - s1) / (r2 - r1)) * (r - r1) + s1
            # Segment III
            elif r2 < (L - 1):
                s = ((L - 1 - s2) / (L - 1 - r2)) * (r - r2) + s2
            else:
                s = s2  # or L-1

            imgout[x, y] = np.uint8(np.clip(s, 0, 255))

    return imgout

def Histogram(imgin):
    M,N =imgin.shape
    imgout=np.zeros((M,L,3),np.uint8) + np.uint8(255)
    h=np.zeros(L,np.int32)
    for x in  range(0,M):
        for y in range(0,N):
            r=imgin[x,y]
            h[r]=h[r]+1
    p=1.0*h/(M*N) 
    scale=3000
    for r in range(0,L):
        cv2.line(imgout,(r,M-1),(r,M-1-np.int32(scale*p[r])),(255,0,0))
    return imgout 
def HistEqual(imgin):
    M, N = imgin.shape
    L = 256  # Ensure L is defined
    imgout = np.zeros((M, N), np.uint8)
    
    # Histogram
    h = np.zeros(L, np.int32)
    for x in range(M):
        for y in range(N):
            r = imgin[x, y]
            h[r] += 1
    
    # Normalize histogram
    p = h / float(M * N)

    # Cumulative distribution function (CDF)
    s = np.zeros(L, np.float64)
    for k in range(L):
        s[k] = np.sum(p[:k+1])

    # Map the old pixel values using CDF
    for x in range(M):
        for y in range(N):
            r = imgin[x, y]
            imgout[x, y] = np.uint8((L - 1) * s[r])
    
    return imgout

def LocalHist(imgin):
    M,N =imgin.shape
    imgout=np.zeros((M,N),np.uint8)
    m = 3
    n = 3
    a = m//3
    b = n//2
    for x in range (a,M-a):
        for y in range(b, M-b):
            w=imgin[x-a:x+a+1, y-b:y+b+1]
            w = cv2.equalizeHist(w)
            imgout[x,y] = w[a,b]
    return imgout
def HistStat(imgin):
    M,N =imgin.shape
    imgout=np.zeros((M,N),np.uint8)

    mean, stddev = cv2.meanStdDev(imgin)
    mG = mean[0,0]
    sigmaG = stddev[0,0]
    m = 3
    n = 3
    a = m // 2
    b = n // 2
    C = 22.8
    k0 = 0.0
    k1 = 0.1
    k2 = 0.0
    k3 = 0.1

    for x in range(a, M-a):

        for y in range(b, N-1):

            w = imgin[x-a:x+a+1, y - b : y + b + 1 ]

            mean, stddev = cv2.meanStdDev (w)

            msxy =  mean [0,0]

            sigmasxy = stddev[0,0]
            if (k0*mG <= msxy <= k1*mG) and (k2*sigmaG <= sigmasxy <= k3*sigmaG):

                imgout [x,y] = np.uint8(C*imgin [x,y])

            else:
                imgout [x,y] = imgin[x,y]
    return imgout
def SmoothBoxfilter(imgin):
    m=21
    n=21
    w=np.zeros((m,n),np.float32)+np.float32(1.0/(m*n))
    imgout = cv2.filter2D(imgin, cv2.CV_8UC1, w)
    return imgout
def SmoothGauss(imgin):
    m=43
    n=43
    sigma=7.0
    a=m//2
    b=n//2
    w=np.zeros((m,n),np.float32)
    for s in range (-a,a+1):
        for t in range (-b,b+1):
            w[s+a,t+b]=np.exp(-(s*s+t*t)/(2*sigma*sigma))
    k=np.sum(w)
    w=w/k
    imgout = cv2.filter2D(imgin, -1, w)
    return imgout
def Sharp(imgin):
    w=np.array([[1,1,1],[1,-8,1],[1,1,1]],np.float32)
    Laplacian=cv2.filter2D(imgin,cv2.CV_32FC1,w)
    imgout=imgin -Laplacian
    imgout=np.clip(imgout,0,L-1)
    imgout=imgout.astype(np.uint8)
    return imgout
# (Các hàm khác tương tự...)

# ===== Giao diện Streamlit =====

st.title("Xử lý ảnh cơ bản với Streamlit")

uploaded_file = st.file_uploader("Tải ảnh lên", type=["jpg", "jpeg", "png","tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)

    # Chuyển sang ảnh xám nếu cần
    img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Danh sách các hàm xử lý ảnh
    functions = {
        "Negative (ảnh xám)": lambda: Negative(img_gray),
        "Negative Color (ảnh màu)": lambda: NegativeColor(img_array),
        "Logarit": lambda: Logarit(img_gray),
        "Power": lambda: Power(img_gray),
        "Piecewise Linear": lambda: PiecewiseLine(img_gray),
        "Histogram (biểu đồ)": lambda: Histogram(img_gray),
        "Histogram Equalization": lambda: HistEqual(img_gray),
        "Local Histogram": lambda: LocalHist(img_gray),
        "Histogram Statistics": lambda: HistStat(img_gray),
        "Smooth (Box Filter)": lambda: SmoothBoxfilter(img_gray),
        "Smooth (Gaussian)": lambda: SmoothGauss(img_gray),
        "Sharpen (Laplacian)": lambda: Sharp(img_gray)
    }

    # ComboBox để chọn chức năng
    option = st.selectbox("Chọn chức năng xử lý ảnh:", list(functions.keys()))

    if st.button("Xử lý ảnh"):
        result = functions[option]()
        st.image(result, caption="Ảnh sau xử lý", use_container_width=True)

# Gợi ý hiển thị ảnh gốc
    with st.expander("Hiển thị ảnh gốc"):
        st.image(image, caption="Ảnh gốc", use_container_width=True)
