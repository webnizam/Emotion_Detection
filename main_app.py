import os
import cv2
import datetime
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from frame_decorator import (
    create_circle_mask,
    draw_arrow,
    draw_circle_frame,
    plot_emotions_circle,
)

# //---------------------Special Vars-----------------------------------
# Change to False to disable image write
switch = True
# Change to False to disable log write
log_switch = False
# //--------------------------------------------------------------------
# #
# //-----------------User Defined Functions-----------------------------
save = False


def write_report(log_switch, label):
    """
    Write to the .csv log
    """
    if log_switch == True:
        with open("camera-log.csv", mode="a") as file:
            file.write(str(label) + " , " + str(datetime.datetime.now()) + "\n")


# //--------------------------------------------------------------------
def render_report(log_switch):
    """
    Read the .csv and write the results to the page in a table
    """
    if log_switch == True:
        st.write("log_switch = " + str(log_switch))
        write_report(log_switch, "Emotion Detection has stopped")
        df = pd.read_csv("camera-log.csv")
        df.columns = ["Emotion", "Date and Time"]
        df.sort_values(by=["Date and Time"], inplace=True, ascending=False)
        st.write(df)


# //--------------------------------------------------------------------
def write_Image(ctr, frame, label, label_position, switch):
    """
    Write the image with label to the corresponding directory
    """
    if switch and save:
        #  If folder doesn't exist, then create it.
        CHECK_FOLDER = os.path.isdir("generated/" + label)
        if not CHECK_FOLDER:
            os.makedirs("generated/" + label)
        cv2.imwrite(
            f"generated/{label}/" + str(ctr) + ".jpg",
            cv2.putText(
                frame,
                label,
                label_position,
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            ),
        )


# //--------------------------------------------------------------------


# Setting Title, Logo and favicon of the Application
logo = "assets/falcons-logo2.png"
favicon = "assets/favicon.ico"
st.set_page_config(page_title="Emotion Detection", page_icon=favicon)
image = Image.open(logo)
st.image(image, caption="Hospitality Emotion Detection by FALCON.AI", width=350)
# Checkbox to turn off/on camera for detection
run = st.checkbox("Check to Run Emotion Detection")

# Font: Montserrat
hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            img {display: flex;
            justify-content: center;}
            code {visibility: hidden;}
            </style>
            """


st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# Set Frame window to camera
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)
# Load emotion classification model
face_classifier = cv2.CascadeClassifier(r"assets/haarcascade_frontalface_default.xml")
classifier = load_model(r"assets/model.h5")
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

emotions = {
    emotion.lower().strip(): ((i * 2) * np.pi) / len(emotion_labels)
    for i, emotion in enumerate(emotion_labels)
}

ctr = 0


while run:
    _, frame1 = camera.read()
    frame = cv2.resize(frame1, (640, 480))
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

    # FRAME_WINDOW.image(frame)
    labels = []

    faces = face_classifier.detectMultiScale(gray)

    try:
        areas = [w * h for x, y, w, h in faces]
        i_biggest = np.argmax(areas)
        x, y, h, w = faces[i_biggest]

        ctr = ctr + 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
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

            # frame = plot_emotions_circle(list(prediction), frame, emotion_labels)

            #  Show analysis in window
            FRAME_WINDOW.image(
                cv2.putText(
                    frame,
                    label,
                    label_position,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
            )

    except Exception as e:
        print(e)
        FRAME_WINDOW.image(
            cv2.putText(
                frame,
                "",
                (0, 0),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        )

        continue
