from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoHTMLAttributes
import streamlit as st
import av
import numpy as np
import pandas as pd
from PIL import Image
from keras.models import load_model
from keras.preprocessing import image
from frame_decorator import create_circle_mask, draw_arrow, draw_circle_frame
from tensorflow.keras.utils import img_to_array
import cv2


logo = "assets/falcons-logo2.png"
favicon = "assets/favicon.ico"
st.set_page_config(page_title="Emotion Detection", page_icon=favicon)
image_ = Image.open(logo)
st.image(image_, caption="Hospitality Emotion Detection by FALCON.AI", width=350)
# Checkbox to turn off/on camera for detection
run = st.checkbox("Check to Run Emotion Detection")

# Font: Montserrat
hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            img {display: flex;
            justify-content: center;}
            body {
                text-align: center;
                align-items: center;
                vertical-align: middle;
                height: 100vh;
                width: 100%;
            }
            code {visibility: hidden;}
            </style>
            """


st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# Set Frame window to camera
FRAME_WINDOW = st.image([])

face_classifier = cv2.CascadeClassifier(r"assets/haarcascade_frontalface_default.xml")
classifier = load_model(r"assets/model.h5")
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

emotions = {
    emotion.lower().strip(): ((i * 2) * np.pi) / len(emotion_labels)
    for i, emotion in enumerate(emotion_labels)
}


# cascade = cv2.CascadeClassifier("assets/haarcascade_frontalface_default.xml")

classifier = load_model(r"assets/model.h5")
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

emotions = {
    emotion.lower().strip(): ((i * 2) * np.pi) / len(emotion_labels)
    for i, emotion in enumerate(emotion_labels)
}


class VideoProcessor:
    def process_image(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = create_circle_mask(frame, (320, 240), 200)
        for label in iter(emotions):
            angle = emotions[label]
            text_offset_x = int(200 * np.cos(angle))
            text_offset_y = int(200 * np.sin(angle))
            text_position = (320 + text_offset_x, 240 + text_offset_y)

            cv2.putText(
                frame,
                label.title(),
                text_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        draw_circle_frame(frame, (320, 240), 200)

        faces = face_classifier.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 3
        )

        try:
            areas = [w * h for x, y, w, h in faces]
            i_biggest = np.argmax(areas)
            x, y, h, w = faces[i_biggest]
            # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y : y + h, x : x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype("float") / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = classifier.predict(roi)[0]
                print(f"{prediction=}")
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)

                angle = emotions[label.lower().strip()]
                # angle = emotions["surprise"]

                print(f"{label=}")
                print(f"{angle=}")

                draw_arrow(frame, angle)

                # cv2.putText(
                #     frame,
                #     label,
                #     label_position,
                #     cv2.FONT_HERSHEY_SIMPLEX,
                #     1,
                #     (0, 255, 0),
                #     2,
                # )

        except Exception as e:
            print(e)
        return frame

    def recv(self, frame_og):
        frame = frame_og.to_ndarray(format="bgr24")
        frame = cv2.resize(frame, (640, 480))
        frame = cv2.flip(frame, 1)
        frame = self.process_image(frame)
        return av.VideoFrame.from_ndarray(frame, format="bgr24")


webrtc_streamer(
    key="key",
    video_processor_factory=VideoProcessor,
    video_html_attrs=VideoHTMLAttributes(
        autoPlay=True, controls=True, style={"width": "100%"}, muted=True
    ),
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
)
