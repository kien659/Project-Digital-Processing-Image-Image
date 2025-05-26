import argparse
import numpy as np
import cv2 as cv
import joblib
import streamlit as st

def Display_KhuonMat_PT():
	st.title("🧒Nhận dạng khuôn mặt 5 người")
def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError
def Main1(video_=''):
    FRAME_WINDOW = st.image([])
    cap = cv.VideoCapture(0 if video_ == '' else video_)

    if 'stop' not in st.session_state:
        st.session_state.stop = False

    if st.button('Stop'):
        st.session_state.stop = True
        cap.release()
    else:
        st.session_state.stop = False

    #if 'frame_stop' not in st.session_state:
     #   st.session_state.frame_stop = cv.imread('./pages/Source/KhuonMat/stop.jpg')

    if st.session_state.stop:
        FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')
        return

    # Load SVC model và thông tin người
    svc = joblib.load('svc.pkl')
    mydict = ['Canh', 'Hoang', 'Hung','Kien','Tri']
    color = [(0, 255, 0), (255, 0, 0), (0, 255, 255),(255,0,255),(255,255,255)]

    # Khởi tạo detector & recognizer
    detector = cv.FaceDetectorYN.create(
         'face_detection_yunet_2023mar.onnx',
        "",
        (320, 320),
        0.9,
        0.3,
        5000
    )
    recognizer = cv.FaceRecognizerSF.create(
        './face_recognition_sface_2021dec.onnx', "")

    tm = cv.TickMeter()
    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        tm.start()
        faces = detector.detect(frame)
        tm.stop()

        if faces[1] is not None:
            for face_info in faces[1]:
                face_align = recognizer.alignCrop(frame, face_info)
                face_feature = recognizer.feature(face_align)
                probs = svc.predict_proba(face_feature)
                max_prob = np.max(probs)
                label = np.argmax(probs)

                threshold = 0.3

                if max_prob >= threshold:
                    name = mydict[label]
                    col = color[label]
                else:
                    name = "Unknown"
                    col = (0, 0, 255)

                coords = face_info[:-1].astype(np.int32)
                cv.rectangle(frame, (coords[0], coords[1]),
                             (coords[0] + coords[2], coords[1] + coords[3]), col, 2)
                cv.putText(frame, name, (coords[0], coords[1] - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

        cv.putText(frame, 'FPS: {:.2f}'.format(tm.getFPS()), (1, 16),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        FRAME_WINDOW.image(frame, channels='BGR')
def Main2():
	c3,c4 = st.columns(2)
	path = "./images/video.mp4"
	with c3:
		video_file = open(path, 'rb')
		video_bytes = video_file.read()
		c3.video(video_bytes)
	with c4:
		Main1(path)
def Main():
	c1,c2 = st.columns(2)
	if c1.button('Play'):
		Main1()
	if c2.button('Video'):
		Main2()
if __name__=="__main__":
	Main()