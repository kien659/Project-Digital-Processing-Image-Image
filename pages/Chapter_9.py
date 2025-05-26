import streamlit as st
import cv2
import numpy as np
from PIL import Image

# --- Các hàm xử lý ảnh ---
L = 256
def Erosion(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
    return cv2.erode(imgin, w)

def dilate(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.dilate(imgin, w)

def Boundary(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    temp = cv2.erode(imgin, w)
    return imgin - temp

def Contour(imgin):
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    Contours, _ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(Contours) == 0:
        return imgout
    contour = Contours[0]
    n = len(contour)
    for i in range(n - 1):
        x1, y1 = contour[i, 0]
        x2, y2 = contour[i + 1, 0]
        cv2.line(imgout, (x1, y1), (x2, y2), (0, 0, 255), 2)
    x1, y1 = contour[-1, 0]
    x2, y2 = contour[0, 0]
    cv2.line(imgout, (x1, y1), (x2, y2), (0, 0, 255), 2)
    return imgout

def ConvexHull(imgin):
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return imgout
    contour = contours[0]
    hull = cv2.convexHull(contour, returnPoints=False)
    n = len(hull)
    for i in range(n - 1):
        p1 = contour[hull[i, 0], 0]
        p2 = contour[hull[i + 1, 0], 0]
        cv2.line(imgout, tuple(p1), tuple(p2), (0, 0, 255), 2)
    p1 = contour[hull[-1, 0], 0]
    p2 = contour[hull[0, 0], 0]
    cv2.line(imgout, tuple(p1), tuple(p2), (0, 0, 255), 2)
    return imgout

def Defectdetect(imgin):
    imgout = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(imgin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return imgout
    contour = contours[0]
    hull = cv2.convexHull(contour, returnPoints=False)
    cv2.drawContours(imgout, [contour], -1, (255, 255, 255), 1)
    defects = cv2.convexityDefects(contour, hull)
    if defects is not None:
        max_depth = np.max(defects[:, :, 3])
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            if d > max_depth // 2:
                x, y = contour[f, 0]
                cv2.circle(imgout, (x, y), 5, (0, 255, 0), -1)
    return imgout

def ConnectedComponents(imgin):
    _, temp = cv2.threshold(imgin, 200, L - 1, cv2.THRESH_BINARY)
    imgout = cv2.medianBlur(temp, 7)
    n, label = cv2.connectedComponents(imgout)
    a = np.bincount(label.flatten())
    for i in range(1, n):
        y = 20 + i * 20
        text = f"co{i:3d} thanh phan lien thong"
    cv2.putText(imgout, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
    return imgout

def Removesmallrice(imgin):
    w = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (81, 81))
    temp = cv2.morphologyEx(imgin, cv2.MORPH_TOPHAT, w)
    _, temp = cv2.threshold(temp, 0, L - 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, label = cv2.connectedComponents(temp)
    a = np.bincount(label.flatten())
    max_value = np.max(a)
    mask = np.isin(label, np.where(a > 0.7 * max_value)[0])
    return (mask * 255).astype(np.uint8)

# --- Giao diện Streamlit ---
st.title("Xử lý ảnh hình thái học")

uploaded_file = st.file_uploader("Tải ảnh grayscale (đen trắng)", type=["png", "jpg", "jpeg", "tif"])

operation = st.selectbox("Chọn phép xử lý", (
    "Erosion",
    "Dilate",
    "Boundary",
    "Contour",
    "Convex Hull",
    "Defect Detection",
    "Connected Components",
    "Remove Small Rice"
))

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if img is None:
        st.error("Ảnh không hợp lệ hoặc không thể đọc.")
    else:
        st.subheader("Ảnh gốc")
        st.image(img, channels="GRAY")

        if operation == "Erosion":
            result = Erosion(img)
        elif operation == "Dilate":
            result = dilate(img)
        elif operation == "Boundary":
            result = Boundary(img)
        elif operation == "Contour":
            result = Contour(img)
        elif operation == "Convex Hull":
            result = ConvexHull(img)
        elif operation == "Defect Detection":
            result = Defectdetect(img)
        elif operation == "Connected Components":
            result = ConnectedComponents(img)
        elif operation == "Remove Small Rice":
            result = Removesmallrice(img)
        else:
            result = img

        st.subheader("Ảnh sau xử lý")
        if len(result.shape) == 2:
            st.image(result, channels="GRAY")
        else:
            st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), channels="RGB")
