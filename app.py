# app.py
import streamlit as st
import cv2
from detect_emotion import detect_emotion

st.title("😃 Real-Time Emotion Detection (Grayscale Model)")
FRAME_WINDOW = st.image([])
run = st.checkbox("Start Webcam")

cap = cv2.VideoCapture(0)
CONFIDENCE_THRESHOLD = 50.0

if run:
    st.success("Webcam started...")
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame.")
            break

        emotions = detect_emotion(frame)

        for res in emotions:
            x, y, w, h = res["box"]
            label = res["label"]
            confidence = res["confidence"]

            if confidence >= CONFIDENCE_THRESHOLD:
                text = f"{label} ({confidence:.1f}%)"
                color = (0, 255, 0)
            else:
                text = "Uncertain"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
else:
    cap.release()
    st.info("Webcam is off.")
